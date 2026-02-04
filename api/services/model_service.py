"""
模型服务 - 单一职责：模型生命周期管理

职责范围：
- 加载模型
- 获取模型信息
- 模型状态管理
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from models.dgi_gin_rf import create_two_stage_dgi_rf


class ModelService:
    """模型服务：负责模型的加载和信息查询"""

    def __init__(self, checkpoint_dir: Optional[str] = None, experiment_name: str = "gnn_dgi_rf_experiment"):
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
        """加载模型"""
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
            
            self.logger.info(f"模型加载成功: experiment={self.experiment_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.model = None
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"error": "模型未加载"}

        performance_metrics = None
        eval_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_eval_results.json")
        if os.path.exists(eval_path):
            try:
                import json
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
                performance_metrics = None

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
