# 区块链AML反洗钱系统API

## 概述

这是一个基于GNN+DGI+随机森林的区块链反洗钱检测系统API，提供交易异常检测服务。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

### 开发模式
```bash
python app.py
```

### 生产模式
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API端点

### 基础URL
```
http://localhost:5000/api/v1
```

### 1. 健康检查
- **URL**: `/health`
- **方法**: GET
- **描述**: 检查API服务状态

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-27T02:30:00",
  "model_loaded": true
}
```

### 2. 预测交易异常
- **URL**: `/predict`
- **方法**: POST
- **描述**: 预测指定交易的异常情况

**请求体**:
```json
{
  "tx_ids": ["tx1", "tx2", "tx3"]
}
```

**响应示例**:
```json
{
  "results": [
    {
      "tx_id": "tx1",
      "prediction": 0,
      "probability": 0.1,
      "is_suspicious": false,
      "confidence": 0.9,
      "timestamp": "2025-11-27T02:30:00"
    }
  ],
  "total_transactions": 1,
  "suspicious_count": 0,
  "timestamp": "2025-11-27T02:30:00"
}
```

### 3. 批量预测
- **URL**: `/batch_predict`
- **方法**: POST
- **描述**: 批量预测整个数据集

**响应示例**:
```json
{
  "results": [...],
  "statistics": {
    "total_transactions": 1000,
    "suspicious_count": 50,
    "legitimate_count": 950,
    "suspicious_rate": 0.05,
    "legitimate_rate": 0.95
  },
  "timestamp": "2025-11-27T02:30:00"
}
```

### 4. 模型信息
- **URL**: `/model/info`
- **方法**: GET
- **描述**: 获取模型信息

**响应示例**:
```json
{
  "model_type": "GNN+DGI+RandomForest",
  "num_features": 165,
  "num_classes": 2,
  "hidden_channels": 128,
  "gnn_layers": 3,
  "rf_n_estimators": 200,
  "rf_max_depth": 15,
  "experiment_name": "gnn_dgi_rf_experiment",
  "checkpoint_dir": "checkpoints",
  "status": "loaded"
}
```

### 5. 加载模型
- **URL**: `/model/load`
- **方法**: POST
- **描述**: 手动加载模型

**响应示例**:
```json
{
  "message": "模型加载成功",
  "timestamp": "2025-11-27T02:30:00"
}
```

### 6. 系统统计
- **URL**: `/statistics`
- **方法**: GET
- **描述**: 获取系统统计信息

**响应示例**:
```json
{
  "system_status": "running",
  "model_loaded": true,
  "timestamp": "2025-11-27T02:30:00",
  "version": "1.0.0"
}
```

### 7. 预测摘要
- **URL**: `/summary`
- **方法**: POST
- **描述**: 获取预测结果摘要

**请求体**:
```json
{
  "results": [...]
}
```

## 错误响应

所有错误都遵循统一格式：

```json
{
  "error": "错误信息",
  "timestamp": "2025-11-27T02:30:00"
}
```

## 使用示例

### Python客户端示例

```python
import requests
import json

# API基础URL
BASE_URL = "http://localhost:5000/api/v1"

# 健康检查
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 预测交易异常
tx_data = {"tx_ids": ["tx1", "tx2", "tx3"]}
response = requests.post(
    f"{BASE_URL}/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps(tx_data)
)
print(response.json())

# 批量预测
response = requests.post(f"{BASE_URL}/batch_predict")
print(response.json())
```

### curl示例

```bash
# 健康检查
curl -X GET http://localhost:5000/api/v1/health

# 预测交易异常
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"tx_ids": ["tx1", "tx2", "tx3"]}'

# 批量预测
curl -X POST http://localhost:5000/api/v1/batch_predict
```

## 配置

### 环境变量
- `FLASK_ENV`: 设置运行环境 (development/production)
- `CHECKPOINT_DIR`: 模型检查点目录 (默认: checkpoints)
- `EXPERIMENT_NAME`: 实验名称 (默认: gnn_dgi_rf_experiment)

### 模型配置
模型配置在`PredictionController`中设置，包括：
- 特征数量: 165
- 类别数量: 2
- 隐藏层维度: 128
- GNN层数: 3
- 随机森林树数量: 200
- 随机森林最大深度: 15

## 注意事项

1. 确保模型文件存在于`checkpoints`目录中
2. 首次启动时会自动加载模型
3. 单次预测交易数量限制为1000个
4. 建议在生产环境中使用Gunicorn运行
5. API支持跨域请求(CORS)

## 故障排除

### 模型加载失败
- 检查`checkpoints`目录是否存在
- 确认模型文件完整性
- 查看日志获取详细错误信息

### 预测失败
- 检查输入数据格式
- 确认模型已正确加载
- 验证交易ID列表不为空