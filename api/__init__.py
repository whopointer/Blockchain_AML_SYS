"""
区块链AML反洗钱系统API模块
提供RESTful API服务用于交易异常检测
"""

from flask import Flask
from flask_cors import CORS
import logging
import os

from .routes import register_routes


def create_app(config_name='development'):
    """
    创建Flask应用

    Args:
        config_name: 配置名称

    Returns:
        Flask: 配置好的Flask应用
    """
    app = Flask(__name__)

    # 启用跨域支持
    CORS(app)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 应用配置
    if config_name == 'development':
        app.config['DEBUG'] = True
    elif config_name == 'production':
        app.config['DEBUG'] = False

    # 创建并加载模型（仅一次）
    from .services.prediction_service import PredictionService
    from .services.model_service import ModelService
    from .services.data_service import DataService
    from .services.cache_service import CacheService
    logger = logging.getLogger(__name__)

    try:
        # 直接使用各子服务加载模型，职责清晰
        model_service = ModelService()
        data_service = DataService()
        cache_service = CacheService()

        logger.info("正在启动时自动加载模型...")
        model_loaded = model_service.load_model()

        if model_loaded:
            logger.info("✅ 模型加载成功，正在加载数据...")
            data_loaded = data_service.load_data()

            if data_loaded:
                logger.info("✅ 数据加载成功，正在构建缓存...")
                cache_built = cache_service.build_cache(model_service.model, data_service.get_full_data())
                if cache_built:
                    logger.info("✅ 缓存构建完成")
                else:
                    logger.warning("⚠️ 缓存构建失败")
            else:
                logger.warning("⚠️ 数据加载失败")
        else:
            logger.warning("⚠️ 模型加载失败，服务将继续运行")
    except Exception as e:
        logger.error(f"❌ 启动时初始化异常: {e}")
        # 降级创建空服务
        model_service = DataService = cache_service = None

    # 创建 PredictionService facade（组合各子服务）
    service = PredictionService(
        model_service=model_service,
        data_service=data_service,
        cache_service=cache_service
    )

    # 将 service 存入 app config，供路由使用
    app.config['PREDICTION_SERVICE'] = service

    # 注册路由（会从 app.config 获取 controller）
    register_routes(app)

    return app


def create_production_app():
    """创建生产环境应用"""
    return create_app('production')


def create_development_app():
    """创建开发环境应用"""
    return create_app('development')


__all__ = [
    'create_app',
    'create_production_app', 
    'create_development_app'
]
