"""
数据库模型定义

使用 SQLAlchemy ORM
"""

from datetime import datetime

from sqlalchemy import (
    create_engine, Column, BigInteger, Integer, String, Text,
    DECIMAL, DateTime, ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .db_config import DATABASE_URL

# 创建引擎
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# 辅助函数
# ============================================================
def init_db():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """删除所有数据库表"""
    Base.metadata.drop_all(bind=engine)
