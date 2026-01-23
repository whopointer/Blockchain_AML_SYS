"""
区块链AML反洗钱系统API应用入口
"""

import os
import sys
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保当前目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from controllers.prediction_controller import PredictionController
from routes import register_routes
from flask import Flask
from flask_cors import CORS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 创建控制器实例
prediction_controller = PredictionController()

# 注册路由
register_routes(app)

# 初始化模型
def initialize_model():
    """初始化模型"""
    try:
        success = prediction_controller.load_model()
        if success:
            logger.info("模型初始化成功")
        else:
            logger.warning("模型初始化失败")
    except Exception as e:
        logger.error(f"模型初始化错误: {e}")


# 初始化模型
initialize_model()


if __name__ == '__main__':
    # 启动应用
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
