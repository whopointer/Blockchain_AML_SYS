"""
统一服务 Facade - 简化版

职责范围：
- 依赖注入（子服务组合）
- 健康检查
- 模型加载

注意：
- 只支持 DGI+GIN+RF 模型
- 从数据库读取特征数据
"""

import logging
from typing import Dict, Any, Optional

from .services.model_factory import ModelFactory
from .services.validation_service import ValidationService


class ServiceFacade:
    """服务 Facade：统一入口，负责依赖注入和健康检查"""

    def __init__(self,
                 checkpoint_dir: Optional[str] = None,
                 experiment_name: str = "gnn_dgi_rf_experiment",
                 model_type: str = "gnn"):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.model_type = model_type.lower()
        self.logger = logging.getLogger(__name__)

        # 验证服务
        self.validation_service = ValidationService()

        # 模型服务 - 只支持 GNN (DGI+GIN+RF)
        self.model_service = ModelFactory.create(
            model_type=self.model_type,
            checkpoint_dir=checkpoint_dir,
            experiment_name=experiment_name
        )

    def initialize(self) -> bool:
        """
        初始化：加载模型
        """
        self.logger.info(f"正在加载 {self.model_type.upper()} 模型...")
        model_loaded = self.model_service.load_model()
        if not model_loaded:
            self.logger.error("模型加载失败")
            return False

        self.logger.info("✅ 服务初始化完成")
        return True

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "status": "healthy" if self.model_service.is_loaded else "unhealthy",
            "model_type": self.model_type,
            "model_loaded": self.model_service.is_loaded,
            "data_loaded": False,
            "cache_built": False,
            "timestamp": self._get_timestamp(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = self.model_service.get_model_info()
        info["current_model_type"] = self.model_type
        return info

    def validate_input(self, tx_ids):
        """验证输入"""
        return self.validation_service.validate_input(tx_ids)

    @staticmethod
    def _get_timestamp() -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


# 全局 facade 实例
_facade: 'ServiceFacade' = None


def get_facade() -> 'ServiceFacade':
    """获取 facade 实例"""
    return _facade


__all__ = ['ServiceFacade', 'get_facade']