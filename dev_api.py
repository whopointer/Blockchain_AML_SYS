#!/usr/bin/env python3
"""
开发模式API启动脚本
"""

import os
import sys

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加api目录到路径
api_dir = os.path.join(project_root, 'api')
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

print("=" * 50)
print("区块链AML系统API服务器 (开发模式)")
print("=" * 50)
print("访问地址: http://localhost:5001")
print("API健康检查: http://localhost:5001/api/v1/health")
print("=" * 50)

# 切换到api目录（确保数据路径正确）
os.chdir(api_dir)

# 导入并启动Flask应用
from api.app import app

if __name__ == '__main__':
    # 设置use_reloader=False避免重启问题
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)