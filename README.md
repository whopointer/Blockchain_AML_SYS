# 区块链AML反洗钱检测系统

基于图神经网络的区块链交易异常检测系统，使用Deep Graph Infomax (DGI) 进行自监督学习，有效识别可疑交易模式。

## 🚀 项目特性

- **图神经网络架构**: 使用改进的GIN (Graph Isomorphism Network) 层进行图表示学习
- **自监督学习**: 集成Deep Graph Infomax进行无监督预训练
- **多尺度特征提取**: 支持多头注意力和多尺度图神经网络
- **异常检测**: 基于节点嵌入的多种异常检测算法 (DBSCAN, KMeans)
- **完整训练流程**: 包含早停、学习率调度、梯度裁剪等高级训练策略
- **全面评估**: 提供ROC曲线、PR曲线、混淆矩阵等评估工具
- **推理引擎**: 支持批量推理、风险评分和相似度分析

## 📁 项目结构

```
blockchain_aml_project/
├───api/                    # API接口 (待开发)
├───config/                 # 配置文件 (待开发)
├───data/                   # 数据处理模块
│   ├───__init__.py        # 数据加载和预处理
│   ├───raw/               # 原始数据
│   ├───data_loader.py     # 高级数据加载器
│   ├───data_utils.py      # 数据处理工具
│   ├───feature_engineering.py # 特征工程
│   └───graph_builder.py   # 图构建器
├───models/                 # 模型定义
│   ├───__init__.py
│   ├───gnn_model.py       # 图神经网络模型
│   ├───dgi.py             # Deep Graph Infomax
│   ├───trainer.py         # 训练器
│   ├───inference.py       # 推理引擎
│   └───evaluator.py       # 模型评估
├───scripts/               # 脚本文件 (待开发)
├───tests/                 # 测试文件 (待开发)
├───run.py                 # 主运行程序
├───requirements.txt       # 依赖包
└───README.md             # 项目说明
```

## 🛠️ 安装说明

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于GPU加速)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd blockchain_aml_project
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **GPU支持 (可选)**
如果您有NVIDIA GPU，可以安装CUDA版本的PyTorch Geometric：

```bash
# 根据您的CUDA版本选择相应的包
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 🚀 快速开始

### 1. 准备Elliptic数据集

将Elliptic数据集文件放置在 `data/raw/` 目录下：
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_features.csv`

### 2. 训练模型

```bash
# 基础训练
python run.py --mode train --epochs 100

# 高级训练配置
python run.py --mode train \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001 \
    --hidden_channels 64 \
    --num_features 165 \
    --num_classes 2
```

### 3. 评估模型

```bash
python run.py --mode eval --model_path checkpoints/model.pth
```

### 4. 推理

```bash
python run.py --mode inference --model_path checkpoints/model.pth
```

## 📊 使用示例

### 基础模型使用

```python
from models.gnn_model import ImprovedGNNModel
from models.dgi import ImprovedDGI
from models.trainer import create_trainer
from data import EllipticDataLoader

# 创建数据加载器
data_loader = EllipticDataLoader('data/')
train_loader = data_loader.get_train_loader(batch_size=32)
val_loader = data_loader.get_val_loader(batch_size=32)

# 创建模型
gnn_model = ImprovedGNNModel(
    num_features=165,      # Elliptic数据集特征数
    num_classes=2,         # 二分类（正常/异常）
    hidden_channels=64,
    use_multi_scale=True,
    use_attention_pooling=True
)

dgi_model = ImprovedDGI(gnn_model, hidden_channels=64)

# 创建训练器
trainer = create_trainer(dgi_model, learning_rate=0.001)

# 训练
results = trainer.train(train_loader, val_loader, num_epochs=100)
```

### 推理和异常检测

```python
from models.inference import create_inference_engine
from data import EllipticDataset

# 创建推理引擎
inference_engine = create_inference_engine(dgi_model)

# 加载数据
dataset = EllipticDataset(root='data/', include_unknown=True)
data = dataset[0]

# 预测节点嵌入
embeddings = inference_engine.predict_node_embeddings(data)

# 异常检测
anomaly_results = inference_engine.detect_anomalies(
    embeddings, method='dbscan', eps=0.5
)

print(f"检测到 {anomaly_results['num_anomalies']} 个异常节点")
```

### 模型评估

```python
from models.evaluator import create_evaluator

# 创建评估器
evaluator = create_evaluator(dgi_model)

# 加载测试数据
test_loader = data_loader.get_test_loader(batch_size=32)

# 评估
metrics = evaluator.evaluate(test_loader)
evaluator.print_metrics(metrics)

# 绘制ROC曲线
evaluator.plot_roc_curve(test_loader, save_path='roc_curve.png')
```

## 🎯 模型架构

### 图神经网络 (GNN)
- **GIN层**: 改进的Graph Isomorphism Network，支持批归一化和残差连接
- **多尺度GNN**: 多头注意力机制，捕获不同尺度的图特征
- **注意力池化**: 智能的图级别特征聚合

### Deep Graph Infomax (DGI)
- **自监督学习**: 无需标签数据学习图表示
- **多种池化策略**: mean, max, add, attention pooling
- **灵活的负采样**: shuffle, negative sampling, feature corruption

### 训练策略
- **早停机制**: 防止过拟合
- **学习率调度**: StepLR, CosineAnnealingLR, ReduceLROnPlateau
- **梯度裁剪**: 稳定训练过程
- **检查点管理**: 自动保存最佳模型

## 📈 评估指标

- **AUC-ROC**: 受试者工作特征曲线下面积
- **AUC-PR**: 精确率-召回率曲线下面积
- **准确率、精确率、召回率、F1分数**
- **混淆矩阵**
- **异常检测指标**: 异常率、聚类质量

## 🔧 配置参数

### 模型参数
- `num_features`: 输入特征维度 (Elliptic数据集为165)
- `num_classes`: 分类类别数
- `hidden_channels`: 隐藏层维度
- `num_layers`: GNN层数
- `dropout`: Dropout概率

### 训练参数
- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `weight_decay`: 权重衰减
- `patience`: 早停耐心值

### DGI参数
- `pooling_strategy`: 池化策略 ('mean', 'max', 'add', 'attention')
- `corruption_method`: 负采样方法 ('shuffle', 'negative_sampling')
- `temperature`: 注意力温度参数

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络库
- [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) - 自监督图学习论文
- [Graph Isomorphism Network](https://arxiv.org/abs/1810.00826) - GIN论文
- [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) - 区块链交易数据集

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至: [1596118915@qq.com]

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！