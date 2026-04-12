"""
区块链AML反洗钱系统API模块
提供RESTful API服务用于交易异常检测

支持模型：
- gnn: DGI+GIN+RandomForest (PyTorch Geometric)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# 全局变量
_facade = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _facade

    logger.info("正在初始化服务...")
    from .facade import ServiceFacade

    # 从环境变量读取配置
    checkpoint_dir = os.environ.get(
        'CHECKPOINT_DIR',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    )
    experiment_name = os.environ.get('EXPERIMENT_NAME', 'gnn_dgi_rf_experiment')

    logger.info(f"初始化 DGI+GIN+RandomForest 模型...")

    _facade = ServiceFacade(
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name
    )
    _facade.initialize()
    logger.info("✅ 服务初始化成功")

    yield

    logger.info("服务已关闭")


def create_app(config_name: str = "development") -> FastAPI:
    """创建 FastAPI 应用
    
    Args:
        config_name: 配置名称 (development / production)
    """
    global logger
    logging.basicConfig(
        level=logging.INFO if config_name == "development" else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    app = FastAPI(
        title="Blockchain AML API",
        description="区块链反洗钱交易异常检测系统 | DGI+GIN+RandomForest",
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config_name == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 延迟导入路由，避免循环导入
    from .routes import api_router
    app.include_router(api_router, prefix="/api/v1")

    return app


def get_facade():
    """获取 facade 实例"""
    from .facade import ServiceFacade
    return _facade


__all__ = ['create_app', 'get_facade']


# 默认应用实例（供 uvicorn api:app 使用）
app = create_app()