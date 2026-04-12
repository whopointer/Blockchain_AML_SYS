"""
GNN 模型服务 - 单一职责：GNN 模型生命周期管理

职责范围：
- 加载 GNN+DGI+RandomForest 模型
- 获取模型信息
- 模型状态管理
- 模型预测（使用子图）
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
from models.dgi_gin_rf import create_two_stage_dgi_rf

# 懒加载 data_loader_service
_data_loader_service = None


def _get_data_loader():
    """获取数据加载服务（懒加载）"""
    global _data_loader_service
    if _data_loader_service is None:
        from .data_loader_service import get_data_loader_service
        _data_loader_service = get_data_loader_service()
    return _data_loader_service


class GNNModelService:
    """GNN 模型服务：负责 GNN 模型的加载和信息查询"""

    def __init__(self, 
                 checkpoint_dir: Optional[str] = None, 
                 experiment_name: str = "gnn_dgi_rf_experiment"):
        
        if checkpoint_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint_dir = os.path.join(project_root, "checkpoints")
        
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._loaded_at: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def load_model(self) -> bool:
        """加载 GNN 模型"""
        try:
            self.model = create_two_stage_dgi_rf(
                num_features=165,
                num_classes=2,
                hidden_channels=128,
                gnn_layers=3,
                rf_n_estimators=200,
                rf_max_depth=15,
                device="auto",
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=self.experiment_name,
            )
            self.model.load_full_model(self.experiment_name)
            self._loaded_at = datetime.now().isoformat()
            
            self.logger.info(f"GNN 模型加载成功: experiment={self.experiment_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"GNN 模型加载失败: {e}")
            self.model = None
            return False

    def predict(self, addresses: List[str], depth: int = 1, max_nodes: int = 500) -> List[Dict[str, Any]]:
        """
        预测一个或多个地址的风险（使用子图）
        
        Args:
            addresses: 地址列表
            depth: 邻居深度（1=直接邻居，2=邻居的邻居）
            max_nodes: 子图最大节点数
            
        Returns:
            预测结果列表，每个元素包含:
                - address: 地址
                - probability: 异常概率
                - is_suspicious: 是否可疑
                - risk_level: 风险等级
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        data_loader = _get_data_loader()
        results = []
        
        for address in addresses:
            # 检查地址是否存在
            if not data_loader.check_address_exists(address):
                results.append({
                    "address": address,
                    "probability": 0.0,
                    "is_suspicious": False,
                    "risk_level": "unknown",
                    "error": "地址不存在于数据库中"
                })
                continue
            
            try:
                # 构建子图
                subgraph = data_loader.build_pyg_subgraph(address, depth, max_nodes)
                
                # 确保目标节点在子图中
                if address not in subgraph.address_to_idx:
                    results.append({
                        "address": address,
                        "probability": 0.0,
                        "is_suspicious": False,
                        "risk_level": "unknown",
                        "error": "地址特征缺失"
                    })
                    continue
                
                # 获取目标节点的索引
                target_idx = subgraph.address_to_idx[address]
                
                # 创建 mask
                num_nodes = subgraph.x.shape[0]
                mask = torch.zeros(num_nodes, dtype=torch.bool)
                mask[target_idx] = True
                
                # 使用模型预测
                pred, proba = self.model.predict_single_graph(
                    subgraph,
                    mask=mask,
                    threshold=0.5
                )
                
                # 获取该节点的预测概率（异常类）
                probability = float(proba[target_idx][1]) if proba is not None else 0.0
                
                # 确定风险等级
                if probability >= 0.7:
                    risk_level = "high"
                elif probability >= 0.4:
                    risk_level = "medium"
                elif probability >= 0.2:
                    risk_level = "low"
                else:
                    risk_level = "normal"
                
                results.append({
                    "address": address,
                    "probability": probability,
                    "is_suspicious": bool(pred[target_idx] == 1),
                    "risk_level": risk_level
                })
                
            except Exception as e:
                self.logger.error(f"预测失败 ({address}): {e}")
                results.append({
                    "address": address,
                    "probability": 0.0,
                    "is_suspicious": False,
                    "risk_level": "error",
                    "error": str(e)
                })
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"error": "模型未加载"}

        performance_metrics = None
        
        # 首先尝试从 summary.json 读取
        summary_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                if 'overall_performance' in summary:
                    perf = summary['overall_performance']
                    performance_metrics = {
                        "accuracy": None,
                        "auc": perf.get('stage2_val_auc'),
                        "average_precision": perf.get('stage2_val_ap'),
                        "threshold": summary.get('stage2', {}).get('optimal_threshold'),
                    }
            except Exception as e:
                self.logger.warning(f"读取性能指标失败: {e}")

        # 尝试从 eval_results.json 读取
        eval_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_eval_results.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_results = json.load(f)
                performance_metrics = {
                    "accuracy": eval_results.get("accuracy"),
                    "precision": eval_results.get("precision"),
                    "recall": eval_results.get("recall"),
                    "f1_score": eval_results.get("f1_score"),
                    "auc": eval_results.get("auc"),
                    "average_precision": eval_results.get("average_precision"),
                    "threshold": eval_results.get("calibrated_threshold"),
                }
            except Exception:
                pass

        return {
            "model_type": "GNN+DGI+RandomForest",
            "model_version": self.experiment_name,
            "loaded_at": self._loaded_at,
            "performance_metrics": performance_metrics,
            "num_features": 165,
            "num_classes": 2,
            "hidden_channels": 128,
            "gnn_layers": 3,
            "rf_n_estimators": 200,
            "rf_max_depth": 15,
            "experiment_name": self.experiment_name,
            "checkpoint_dir": self.checkpoint_dir,
            "status": "loaded" if self.is_loaded else "not_loaded",
        }


__all__ = ['GNNModelService']
