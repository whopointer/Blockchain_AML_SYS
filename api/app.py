"""
区块链AML反洗钱系统API应用入口
"""

import logging

from api import create_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用（工厂）
app = create_app('development')

if __name__ == '__main__':
    # 启动应用
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
