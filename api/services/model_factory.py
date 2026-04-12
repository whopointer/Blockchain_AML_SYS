"""
模型工厂 - 单一职责：创建模型服务实例

职责范围：
- 根据 model_type 创建对应的模型服务
- 统一管理支持的模型类型

支持模型：
- gnn: GNN+DGI+RandomForest (PyTorch Geometric)
"""

import logging
from typing import Dict, Any, Optional

from .gnn_model_service import GNNModelService


logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂：根据 model_type 创建对应的模型服务"""

    SUPPORTED_MODELS: Dict[str, str] = {
        "gnn": "GNN+DGI+RandomForest (PyTorch Geometric)",
    }

    @classmethod
    def create(cls, 
               model_type: str, 
               checkpoint_dir: Optional[str] = None,
               experiment_name: Optional[str] = None) -> Any:
        """
        创建模型服务实例
        
        Args:
            model_type: 模型类型 ('gnn')
            checkpoint_dir: checkpoint 目录
            experiment_name: 实验名称
            
        Returns:
            GNNModelService 实例
        """
        model_type = model_type.lower()
        
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的模型类型: {model_type}，支持的类型: {list(cls.SUPPORTED_MODELS.keys())}"
            )
        
        # 设置默认 experiment_name
        if experiment_name is None:
            experiment_name = f"{model_type}_experiment"
        
        if model_type == "gnn":
            return GNNModelService(
                checkpoint_dir=checkpoint_dir,
                experiment_name=experiment_name
            )
        
        raise ValueError(f"不支持的模型类型: {model_type}")

    @classmethod
    def get_supported_models(cls) -> Dict[str, str]:
        """获取支持的模型列表"""
        return cls.SUPPORTED_MODELS.copy()


__all__ = ['ModelFactory']