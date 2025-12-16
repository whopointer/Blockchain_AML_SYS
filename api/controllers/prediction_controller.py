"""
预测控制器
处理交易异常预测相关的业务逻辑
"""

import logging
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
from torch_geometric.data import Data, DataLoader
from pathlib import Path

from models.two_stage_dgi_rf import create_two_stage_dgi_rf
from data import EllipticDataset


class PredictionController:
    """预测控制器类"""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints', experiment_name: str = 'gnn_dgi_rf_experiment'):
        """
        初始化预测控制器
        
        Args:
            checkpoint_dir: 检查点目录
            experiment_name: 实验名称
        """
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.dataset = None
        self.tx_mapping = {}
        
    def load_model(self) -> bool:
        """
        加载预训练模型
        
        Returns:
            bool: 是否加载成功
        """
        try:
            self.model = create_two_stage_dgi_rf(
                num_features=165,
                num_classes=2,
                hidden_channels=128,
                gnn_layers=3,
                rf_n_estimators=200,
                rf_max_depth=15,
                device='auto',
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=self.experiment_name
            )
            self.model.load_full_model(self.experiment_name)
            
            # 加载数据集以建立交易ID映射
            self._load_dataset_mapping()
            
            self.logger.info("模型加载成功")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def _load_dataset_mapping(self):
        """加载数据集和交易ID映射"""
        try:
            # 加载完整数据集
            self.dataset = EllipticDataset(
                root='data/raw',
                include_unknown=False
            )
            
            # 创建交易ID到数据索引的映射
            if hasattr(self.dataset, 'merged_df') and 'txId' in self.dataset.merged_df.columns:
                self.tx_mapping = {
                    tx_id: idx for idx, tx_id in enumerate(self.dataset.merged_df['txId'])
                }
                self.logger.info(f"已加载 {len(self.tx_mapping)} 个交易ID映射")
            else:
                self.logger.warning("无法创建交易ID映射")
                
        except Exception as e:
            self.logger.error(f"加载数据集映射失败: {e}")
    
    def predict_transactions(self, tx_ids: List[str]) -> List[Dict[str, Any]]:
        """
        预测指定交易的异常情况
        
        Args:
            tx_ids: 交易ID列表
            
        Returns:
            List[Dict]: 预测结果列表
        """
        if not self.model:
            raise ValueError("模型未加载")
        
        if not self.dataset or not self.tx_mapping:
            raise ValueError("数据集未加载，无法进行交易预测")
        
        results = []
        
        try:
            # 查找交易ID对应的数据索引
            found_indices = []
            found_tx_ids = []
            
            for tx_id in tx_ids:
                if tx_id in self.tx_mapping:
                    found_indices.append(self.tx_mapping[tx_id])
                    found_tx_ids.append(tx_id)
                else:
                    # 交易ID未找到，返回默认结果
                    results.append({
                        'tx_id': tx_id,
                        'prediction': 0,
                        'probability': 0.0,
                        'is_suspicious': False,
                        'confidence': 0.0,
                        'risk_level': 'low',
                        'timestamp': datetime.now().isoformat(),
                        'error': '交易ID未找到'
                    })
            
            if found_indices:
                # 从数据集中提取特定交易的数据
                subset_data = self._extract_subset_data(found_indices)
                
                if subset_data:
                    # 创建数据加载器
                    data_loader = DataLoader([subset_data], batch_size=1, shuffle=False)
                    
                    # 执行预测
                    predictions, probabilities = self.model.predict(data_loader)
                    
                    # 构建结果
                    for i, tx_id in enumerate(found_tx_ids):
                        pred = int(predictions[i])
                        prob = float(probabilities[i, 1])
                        confidence = float(max(prob, 1 - prob))
                        
                        # 确定风险级别
                        if prob > 0.7:
                            risk_level = 'high'
                        elif prob > 0.4:
                            risk_level = 'medium'
                        else:
                            risk_level = 'low'
                        
                        results.append({
                            'tx_id': tx_id,
                            'prediction': pred,
                            'probability': prob,
                            'is_suspicious': bool(pred == 1),
                            'confidence': confidence,
                            'risk_level': risk_level,
                            'timestamp': datetime.now().isoformat()
                        })
            
        except Exception as e:
            self.logger.error(f"预测过程出错: {e}")
            # 为所有交易ID返回错误结果
            for tx_id in tx_ids:
                if not any(r['tx_id'] == tx_id for r in results):
                    results.append({
                        'tx_id': tx_id,
                        'prediction': 0,
                        'probability': 0.0,
                        'is_suspicious': False,
                        'confidence': 0.0,
                        'risk_level': 'low',
                        'timestamp': datetime.now().isoformat(),
                        'error': '预测过程出错'
                    })
        
        return results
    
    def _extract_subset_data(self, indices: List[int]) -> Data:
        """从数据集中提取指定索引的子集数据"""
        try:
            # 获取完整图数据
            full_data = self.dataset[0]  # EllipticDataset返回单个图
            
            # 提取指定节点的特征
            node_features = full_data.x[indices]
            node_labels = full_data.y[indices] if full_data.y is not None else None
            
            # 创建节点索引映射
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
            
            # 提取相关的边
            edge_index = full_data.edge_index
            subset_edges = []
            
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if src in old_to_new and dst in old_to_new:
                    subset_edges.append([old_to_new[src], old_to_new[dst]])
            
            # 转换为张量
            if subset_edges:
                new_edge_index = torch.tensor(subset_edges, dtype=torch.long).t().contiguous()
            else:
                # 如果没有边，创建自环
                new_edge_index = torch.tensor([[i, i] for i in range(len(indices))], dtype=torch.long).t().contiguous()
            
            # 创建子图数据
            subset_data = Data(
                x=node_features,
                edge_index=new_edge_index,
                y=node_labels,
                num_nodes=len(indices)
            )
            
            return subset_data
            
        except Exception as e:
            self.logger.error(f"提取子集数据失败: {e}")
            return None
    
    def batch_predict(self) -> Dict[str, Any]:
        """
        批量预测整个数据集
        
        Returns:
            Dict: 批量预测结果
        """
        if not self.model:
            raise ValueError("模型未加载")
        
        try:
            self.logger.info("开始批量推理...")
            
            # 使用完整数据集
            if not self.dataset:
                self.dataset = EllipticDataset(
                    root='data/raw',
                    include_unknown=False
                )
            
            # 创建数据加载器
            data_loader = DataLoader([self.dataset[0]], batch_size=1, shuffle=False)
            
            # 执行预测
            predictions, probabilities = self.model.predict(data_loader)
            
            # 获取交易ID
            if hasattr(self.dataset, 'merged_df') and 'txId' in self.dataset.merged_df.columns:
                tx_ids = self.dataset.merged_df['txId'].tolist()
            else:
                tx_ids = [f'tx_{i}' for i in range(len(predictions))]
            
            # 构建结果
            results = []
            for i, tx_id in enumerate(tx_ids):
                pred = int(predictions[i])
                prob = float(probabilities[i, 1])
                confidence = float(max(prob, 1 - prob))
                
                # 确定风险级别
                if prob > 0.7:
                    risk_level = 'high'
                elif prob > 0.4:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                
                results.append({
                    'tx_id': tx_id,
                    'prediction': pred,
                    'probability': prob,
                    'is_suspicious': bool(pred == 1),
                    'confidence': confidence,
                    'risk_level': risk_level
                })
            
            # 计算统计信息
            total_transactions = len(predictions)
            suspicious_count = int(np.sum(predictions))
            legitimate_count = total_transactions - suspicious_count
            
            return {
                'results': results,
                'statistics': {
                    'total_transactions': total_transactions,
                    'suspicious_count': suspicious_count,
                    'legitimate_count': legitimate_count,
                    'suspicious_rate': float(suspicious_count / total_transactions),
                    'legitimate_rate': float(legitimate_count / total_transactions)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"批量预测错误: {e}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        if not self.model:
            return {'error': '模型未加载'}
        
        return {
            'model_type': 'GNN+DGI+RandomForest',
            'num_features': 165,
            'num_classes': 2,
            'hidden_channels': 128,
            'gnn_layers': 3,
            'rf_n_estimators': 200,
            'rf_max_depth': 15,
            'experiment_name': self.experiment_name,
            'checkpoint_dir': self.checkpoint_dir,
            'status': 'loaded'
        }
    
    def validate_input(self, tx_ids: List[str]) -> Tuple[bool, str]:
        """
        验证输入数据
        
        Args:
            tx_ids: 交易ID列表
            
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not tx_ids:
            return False, "交易ID列表不能为空"
        
        if not isinstance(tx_ids, list):
            return False, "交易ID必须是列表格式"
        
        if len(tx_ids) > 1000:
            return False, "单次预测交易数量不能超过1000"
        
        for tx_id in tx_ids:
            if not isinstance(tx_id, str):
                return False, "交易ID必须是字符串格式"
            if len(tx_id) == 0:
                return False, "交易ID不能为空字符串"
        
        return True, ""
    
    def get_prediction_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取预测结果摘要
        
        Args:
            results: 预测结果列表
            
        Returns:
            Dict: 预测摘要
        """
        if not results:
            return {'error': '没有预测结果'}
        
        total = len(results)
        suspicious = sum(1 for r in results if r['is_suspicious'])
        legitimate = total - suspicious
        
        # 计算平均置信度
        avg_confidence = sum(r['confidence'] for r in results) / total
        
        # 计算风险分布
        high_risk = sum(1 for r in results if r['probability'] > 0.8)
        medium_risk = sum(1 for r in results if 0.5 <= r['probability'] <= 0.8)
        low_risk = total - high_risk - medium_risk
        
        return {
            'total_transactions': total,
            'suspicious_count': suspicious,
            'legitimate_count': legitimate,
            'suspicious_rate': float(suspicious / total),
            'average_confidence': float(avg_confidence),
            'risk_distribution': {
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            },
            'timestamp': datetime.now().isoformat()
        }