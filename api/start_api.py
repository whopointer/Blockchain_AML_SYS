#!/usr/bin/env python3
"""
区块链AML系统API启动脚本
用于启动REST API服务
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
        default=5001,
        help='服务器端口 (默认: 5001)'
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
        help='工作进程数 (默认: 1)'
    )
    return parser.parse_args()


def start_development_server(args):
    """启动开发服务器"""
    logger.info("启动开发服务器...")
    
    # 导入Flask应用
    try:
        from api.app import app
    except ImportError as e:
        logger.error(f"导入API应用失败: {e}")
        logger.info("尝试从当前目录导入...")
        import sys
        import os
        api_dir = os.path.join(os.getcwd(), 'api')
        if api_dir not in sys.path:
            sys.path.insert(0, api_dir)
        from app import app
    
    # 设置配置
    app.config['CHECKPOINT_DIR'] = args.checkpoint_dir
    app.config['EXPERIMENT_NAME'] = args.experiment_name
    
    # 启动服务器
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


def start_production_server(args):
    """启动生产服务器"""
    logger.info("启动生产服务器...")
    
    import subprocess
    
    # 构建gunicorn命令
    cmd = [
        'gunicorn',
        '--bind', f'{args.host}:{args.port}',
        '--workers', str(args.workers),
        '--timeout', '300',
        '--keep-alive', '5',
        '--max-requests', '1000',
        '--max-requests-jitter', '100',
        '--access-logfile', 'logs/api_access.log',
        '--error-logfile', 'logs/api_error.log',
        'api.app:app'
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env['CHECKPOINT_DIR'] = args.checkpoint_dir
    env['EXPERIMENT_NAME'] = args.experiment_name
    
    # 启动服务器
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
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 50)
    logger.info("区块链AML系统API服务器")
    logger.info("=" * 50)
    logger.info(f"主机: {args.host}")
    logger.info(f"端口: {args.port}")
    logger.info(f"调试模式: {args.debug}")
    logger.info(f"检查点目录: {args.checkpoint_dir}")
    logger.info(f"实验名称: {args.experiment_name}")
    logger.info("=" * 50)
    
    try:
        if args.debug:
            start_development_server(args)
        else:
            start_production_server(args)
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()