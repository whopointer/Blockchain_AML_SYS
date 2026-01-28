"""
日志配置模块
提供统一的日志设置和配置功能
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_name: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志文件保存目录
        log_name: 日志文件名前缀，如果为None则使用时间戳
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
    
    Returns:
        配置好的logger对象
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"blockchain_aml_{timestamp}.log"
    else:
        log_filename = f"{log_name}.log"
    
    log_file_path = os.path.join(log_dir, log_filename)
    
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建logger
    logger = logging.getLogger("blockchain_aml")
    logger.setLevel(numeric_level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加文件处理器
    if file_output:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"日志系统初始化完成，日志文件: {log_file_path}")
    
    return logger


def get_logger(name: str = "blockchain_aml") -> logging.Logger:
    """
    获取logger对象
    
    Args:
        name: logger名称
    
    Returns:
        logger对象
    """
    return logging.getLogger(name)


def setup_mode_logger(
    mode: str,
    experiment_name: str,
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> logging.Logger:
    """
    按运行模式设置专用日志
    
    Args:
        mode: 运行模式 ("training", "evaluation", "inference")
        experiment_name: 实验名称
        log_dir: 日志根目录
        log_level: 日志级别
    
    Returns:
        配置好的模式logger
    """
    # 根据模式确定子目录
    mode_dir_map = {
        "training": "training",
        "eval": "evaluation", 
        "evaluation": "evaluation",
        "inference": "inference"
    }
    
    if mode not in mode_dir_map:
        raise ValueError(f"不支持的运行模式: {mode}，支持的模式: {list(mode_dir_map.keys())}")
    
    # 构建模式专用目录
    mode_log_dir = os.path.join(log_dir, mode_dir_map[mode])
    os.makedirs(mode_log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{mode}_{experiment_name}_{timestamp}"
    
    return setup_logging(
        log_level=log_level,
        log_dir=mode_log_dir,
        log_name=log_name,
        console_output=True,
        file_output=True
    )


def setup_training_logger(
    experiment_name: str,
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> logging.Logger:
    """
    设置训练专用日志
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        log_level: 日志级别
    
    Returns:
        配置好的训练logger
    """
    return setup_mode_logger(
        mode="training",
        experiment_name=experiment_name,
        log_dir=log_dir,
        log_level=log_level
    )


def setup_inference_logger(
    experiment_name: str,
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> logging.Logger:
    """
    设置推理专用日志
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        log_level: 日志级别
    
    Returns:
        配置好的推理logger
    """
    return setup_mode_logger(
        mode="inference",
        experiment_name=experiment_name,
        log_dir=log_dir,
        log_level=log_level
    )


def setup_evaluation_logger(
    experiment_name: str,
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> logging.Logger:
    """
    设置评估专用日志
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        log_level: 日志级别
    
    Returns:
        配置好的评估logger
    """
    return setup_mode_logger(
        mode="evaluation",
        experiment_name=experiment_name,
        log_dir=log_dir,
        log_level=log_level
    )


def log_system_info(logger: logging.Logger):
    """
    记录系统信息
    
    Args:
        logger: logger对象
    """
    import sys
    import platform
    import torch
    
    logger.info("=" * 50)
    logger.info("系统信息")
    logger.info("=" * 50)
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        logger.info("CUDA不可用，使用CPU训练")
    
    logger.info("=" * 50)


def log_model_info(logger: logging.Logger, model, device):
    """
    记录模型信息
    
    Args:
        logger: logger对象
        model: 模型对象
        device: 设备对象
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 50)
    logger.info("模型信息")
    logger.info("=" * 50)
    logger.info(f"模型类型: {type(model).__name__}")
    logger.info(f"训练设备: {device}")
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    logger.info(f"模型大小: {total_params * 4 / 1e6:.2f} MB (假设float32)")
    logger.info("=" * 50)


# 默认配置
DEFAULT_LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'log_dir': 'logs'
}


__all__ = [
    'setup_logging',
    'get_logger', 
    'setup_mode_logger',
    'setup_training_logger',
    'setup_evaluation_logger',
    'setup_inference_logger',
    'log_system_info',
    'log_model_info',
    'DEFAULT_LOG_CONFIG'
]