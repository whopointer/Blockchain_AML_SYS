"""
区块链AML反洗钱系统API模块
提供RESTful API服务用于交易异常检测
"""

from flask import Flask
from flask_cors import CORS
import logging
import os

from routes import register_routes


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
