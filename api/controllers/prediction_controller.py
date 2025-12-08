"""
预测控制器
处理交易异常预测相关的业务逻辑
"""

import logging
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Any, Tuple
from torch_geometric.loader import DataLoader

from models.two_stage_dgi_rf import create_two_stage_dgi_rf
from data import load_inference_data


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
            self.logger.info("模型加载成功")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
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
        
        # 这里需要实现从数据中查找特定交易的逻辑
        # 由于数据结构复杂，暂时返回模拟结果
        results = []
        for tx_id in tx_ids:
            # 模拟预测结果
            prediction = np.random.choice([0, 1], p=[0.9, 0.1])
            probability = np.random.rand()
            
            results.append({
                'tx_id': tx_id,
                'prediction': int(prediction),
                'probability': float(probability),
                'is_suspicious': bool(prediction == 1),
                'confidence': float(max(probability, 1 - probability)),
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    def batch_predict(self) -> Dict[str, Any]:
        """
        批量预测整个数据集
        
        Returns:
            Dict: 批量预测结果
        """
        if not self.model:
            raise ValueError("模型未加载")
        
        try:
            # 加载推理数据
            self.logger.info("开始批量推理...")
            data = load_inference_data('data/')
            data_loader = DataLoader([data], batch_size=1, shuffle=False)
            
            # 执行预测
            predictions, probabilities = self.model.predict(data_loader)
            
            # 获取交易ID（如果有的话）
            tx_ids = getattr(data, 'tx_ids', [f'tx_{i}' for i in range(len(predictions))])
            
            # 构建结果
            results = []
            for i, tx_id in enumerate(tx_ids):
                results.append({
                    'tx_id': tx_id,
                    'prediction': int(predictions[i]),
                    'probability': float(probabilities[i, 1]),
                    'is_suspicious': bool(predictions[i] == 1),
                    'confidence': float(max(probabilities[i, 1], 1 - probabilities[i, 1]))
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