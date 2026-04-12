#!/usr/bin/env python3
"""
区块链AML系统API启动脚本
用于启动FastAPI REST API服务
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """创建必要的目录"""
    directories = ['logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='区块链AML系统API服务器')
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器主机地址 (默认: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='服务器端口 (默认: 8000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='启用热重载 (开发模式)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='模型检查点目录 (默认: checkpoints)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='gnn_dgi_rf_experiment',
        help='实验名称 (默认: gnn_dgi_rf_experiment)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='工作进程数 (生产模式，默认: 1)'
    )
    return parser.parse_args()


def start_server(args):
    """启动服务器"""
    import uvicorn
    
    # 设置环境变量（供 facade 使用）
    os.environ['CHECKPOINT_DIR'] = args.checkpoint_dir
    os.environ['EXPERIMENT_NAME'] = args.experiment_name
    
    config = uvicorn.Config(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=True,
    )
    
    server = uvicorn.Server(config)
    logger.info(f"启动服务器: http://{args.host}:{args.port}")
    if args.reload:
        logger.info("热重载已启用")
    server.run()


def start_production_server(args):
    """启动生产服务器 (使用 gunicorn + uvicorn workers)"""
    import subprocess
    
    # 构建 gunicorn 命令
    cmd = [
        'gunicorn',
        '--bind', f'{args.host}:{args.port}',
        '--workers', str(args.workers),
        '--worker-class', 'uvicorn.workers.UvicornWorker',
        '--timeout', '300',
        '--keep-alive', '5',
        '--max-requests', '1000',
        '--max-requests-jitter', '100',
        '--access-logfile', 'logs/api_access.log',
        '--error-logfile', 'logs/api_error.log',
        '--capture-output',
        'api:app'
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env['CHECKPOINT_DIR'] = args.checkpoint_dir
    env['EXPERIMENT_NAME'] = args.experiment_name
    
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"服务器启动失败，返回码: {result.returncode}")
            logger.error(f"标准输出: {result.stdout}")
            logger.error(f"错误输出: {result.stderr}")
        else:
            logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建必要目录
    create_directories()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    logger.info("=" * 50)
    logger.info("区块链AML系统 FastAPI 服务器")
    logger.info("=" * 50)
    logger.info(f"主机: {args.host}")
    logger.info(f"端口: {args.port}")
    logger.info(f"热重载: {args.reload}")
    logger.info(f"调试模式: {args.debug}")
    logger.info(f"检查点目录: {args.checkpoint_dir}")
    logger.info(f"实验名称: {args.experiment_name}")
    logger.info("=" * 50)
    logger.info("API 文档: http://localhost:8000/docs")
    logger.info("=" * 50)
    
    try:
        if args.workers > 1 or args.debug:
            start_production_server(args)
        else:
            start_server(args)
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
