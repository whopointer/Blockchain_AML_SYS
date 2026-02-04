"""
区块链AML反洗钱系统API模块
提供RESTful API服务用于交易异常检测
"""

from flask import Flask
from flask_cors import CORS
import logging
import os

from .routes import register_routes
from .facade import ServiceFacade


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
    logger = logging.getLogger(__name__)

    # 应用配置
    if config_name == 'development':
        app.config['DEBUG'] = True
    elif config_name == 'production':
        app.config['DEBUG'] = False

    # 创建 ServiceFacade（依赖注入）
    facade = ServiceFacade(
        checkpoint_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints"
        ),
        experiment_name="gnn_dgi_rf_experiment"
    )

    # 启动时初始化（加载模型 → 加载数据 → 构建缓存）
    logger.info("正在初始化服务...")
    initialized = facade.initialize()
    if initialized:
        logger.info("✅ 服务初始化成功")
    else:
        logger.warning("⚠️ 服务初始化失败，服务将继续运行")

    # 将 facade 存入 app config，供路由使用
    app.config['SERVICE_FACADE'] = facade

    # 注册路由
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