"""
数据库配置

从环境变量读取配置，支持本地开发和 Docker 部署
"""

import os
from typing import Optional


def get_env(key: str, default: str = "") -> str:
    """获取环境变量"""
    return os.environ.get(key, default)


# 数据库配置
DB_HOST = get_env("DB_HOST", "localhost")
DB_PORT = int(get_env("DB_PORT", "3306"))
DB_NAME = get_env("DB_NAME", "DataProcessing")
DB_USER = get_env("DB_USER", "root")
DB_PASSWORD = get_env("DB_PASSWORD", "123456")

# 构建数据库 URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# API 配置
API_KEY = get_env("API_KEY", "your_api_key")


class DatabaseConfig:
    """数据库配置类"""

    def __init__(self):
        self.host = DB_HOST
        self.port = DB_PORT
        self.database = DB_NAME
        self.user = DB_USER
        self.password = DB_PASSWORD
        self.url = DATABASE_URL

    def __repr__(self):
        return f"DatabaseConfig(host={self.host}, port={self.port}, database={self.database})"