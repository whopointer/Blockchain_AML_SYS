"""
统一服务 Facade

职责范围：
- 依赖注入（子服务组合）
- 健康检查
- 初始化编排

不负责：
- 预测推理逻辑（由 prediction_service.py 负责）
- 模型管理（由 model_service.py 负责）
- 数据管理（由 data_service.py 负责）
- 缓存管理（由 cache_service.py 负责）
"""

import logging
from typing import Dict, Any, Optional

from .services.model_service import ModelService
from .services.data_service import DataService
from .services.cache_service import CacheService
from .services.validation_service import ValidationService
from .services.prediction_service import PredictionService


class ServiceFacade:
    """服务 Facade：统一入口，负责依赖注入和健康检查"""

    def __init__(self,
                 checkpoint_dir: Optional[str] = None,
                 experiment_name: str = "gnn_dgi_rf_experiment"):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)

        # 子服务由 facade 统一创建和管理
        self.model_service = ModelService(checkpoint_dir, experiment_name)
        self.data_service = DataService()
        self.cache_service = CacheService(experiment_name)
        self.validation_service = ValidationService()

        # 预测服务依赖子服务
        self.prediction_service = PredictionService(
            model_service=self.model_service,
            data_service=self.data_service,
            cache_service=self.cache_service,
            validation_service=self.validation_service
        )

    def initialize(self) -> bool:
        """初始化：加载模型 → 加载数据 → 构建缓存"""
        # 1. 加载模型
        self.logger.info("正在加载模型...")
        model_loaded = self.model_service.load_model()
        if not model_loaded:
            self.logger.error("模型加载失败")
            return False

        # 2. 加载数据
        self.logger.info("正在加载数据...")
        data_loaded = self.data_service.load_data()
        if not data_loaded:
            self.logger.error("数据加载失败")
            return False

        # 3. 构建缓存
        self.logger.info("正在构建缓存...")
        full_data = self.data_service.get_full_data()
        if full_data is not None:
            cache_built = self.cache_service.build_cache(
                self.model_service.model, 
                full_data
            )
            if not cache_built:
                self.logger.warning("缓存构建失败")
        else:
            self.logger.warning("无法获取完整数据，缓存构建跳过")

        self.logger.info("✅ 服务初始化完成")
        return True

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "status": "healthy" if self.model_service.is_loaded else "unhealthy",
            "model_loaded": self.model_service.is_loaded,
            "data_loaded": self.data_service.is_loaded,
            "cache_built": self.cache_service.is_cache_built,
            "timestamp": self._get_timestamp(),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_service.get_model_info()

    def validate_input(self, tx_ids):
        """验证输入"""
        return self.validation_service.validate_input(tx_ids)

    @staticmethod
    def _get_timestamp() -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


__all__ = ['ServiceFacade']
