# 区块链AML反洗钱检测系统

基于两阶段DGI+GIN+随机森林的区块链交易异常检测系统

## 📋 项目概述

本项目实现了一个先进的区块链反洗钱(AML)检测系统，采用"两阶段"模型架构：

**第一阶段**：使用深度图信息最大化(DGI)算法结合图同构网络(GIN)编码器进行自监督学习，学习比特币交易图中每个交易节点的低维嵌入表示。该阶段**包含unknown节点**，用于提升表示质量。

**第二阶段**：将学习到的节点嵌入与原始特征拼接，训练随机森林分类器进行监督的异常交易检测。该阶段**过滤unknown标签**，避免噪声干扰监督学习。

## 🏗️ 系统架构

### 核心模型架构

```
第一阶段：自监督图表示学习
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   交易图数据    │───▶│   GIN编码器      │───▶│   DGI判别器     │
│ (节点+边+特征)  │    │ (图同构网络)     │    │ (对比学习)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    节点嵌入表示 (128维)

第二阶段：监督分类检测
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   节点嵌入      │───▶│   特征拼接       │───▶│  随机森林分类器  │
│   (128维)       │    │(嵌入+原始特征)   │    │ (二分类)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 技术特点

- **GIN编码器**：具有1-WL同构判别能力，能区分不同拓扑结构的子图
- **对比学习机制**：通过特征扰动生成负样本，最大化局部-全局互信息
- **特征融合**：结合图结构信息(嵌入)和原始节点属性
- **阈值校准**：eval阶段基于测试集分布重校准阈值，避免跨集阈值失配
- **API全图推理**：API使用全图Data构造，避免时间步子图索引错位

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- scikit-learn
- pandas, numpy

### 安装依赖

```bash
pip install torch torch-geometric scikit-learn pandas numpy
```

### 数据准备

将Elliptic数据集放在`data/raw/`目录下：
- `elliptic_txs_classes.csv` - 交易类别标签
- `elliptic_txs_edgelist.csv` - 交易边列表
- `elliptic_txs_features.csv` - 交易特征

### 训练模型

#### 两阶段训练（推荐）

```bash
# 基础训练（100轮DGI预训练）
python3 run.py --mode gnn_dgi_rf --dgi_epochs 100

# 带超参数调优的训练
python3 run.py --mode gnn_dgi_rf --dgi_epochs 100 --rf_hyperparameter_tuning

# 自定义参数训练
python3 run.py --mode gnn_dgi_rf \
  --dgi_epochs 150 \
  --hidden_channels 256 \
  --gnn_layers 4 \
  --rf_n_estimators 300 \
  --experiment_name custom_experiment
```

#### 模型评估（包含阈值校准）

```bash
python3 run.py --mode eval
```

评估结果会写入：
- `checkpoints/gnn_dgi_rf_experiment_eval_results.json`
- 包含 `calibrated_threshold`

#### 推理预测

```bash
python3 run.py --mode inference
```

## 📊 模型配置

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_features` | 165 | 输入特征维度 |
| `hidden_channels` | 128 | GIN隐藏层维度 |
| `gnn_layers` | 3 | GIN层数 |
| `dgi_epochs` | 100 | DGI预训练轮数 |
| `rf_n_estimators` | 200 | 随机森林树数量 |
| `rf_max_depth` | 15 | 随机森林最大深度 |

### 训练参数

```bash
python3 run.py --mode gnn_dgi_rf \
  --epochs 200 \
  --batch_size 64 \
  --lr 0.001 \
  --patience 20 \
  --device cuda \
  --checkpoint_dir checkpoints \
  --experiment_name experiment_001
```

## 📁 项目结构

```
Blockchain_AML_SYS/
├── data/                   # 数据处理模块
│   ├── data_loader.py      # 数据加载器
│   ├── feature_engineering.py  # 特征工程
│   └── raw/               # 原始数据
├── models/                 # 模型定义
│   ├── dgi.py            # DGI+GIN模型
│   ├── two_stage_dgi_rf.py  # 两阶段训练模型
│   ├── random_forest_classifier.py  # 随机森林分类器
│   └── trainer.py        # 训练器
├── api/                    # REST API服务
│   ├── app.py            # Flask应用
│   ├── controllers/      # 控制器
│   ├── routes.py         # 路由定义
│   └── schemas/          # 数据验证
├── checkpoints/            # 模型检查点
├── logs/                   # 训练日志
└── run.py                 # 主训练脚本
```

## 🔧 API服务

### 启动API服务

```bash
cd api
python app.py
```

### API端点

- `GET /api/v1/health` - 健康检查
- `POST /api/v1/predict` - 预测指定交易
- `POST /api/v1/batch_predict` - 批量预测
- `GET /api/v1/model/info` - 获取模型信息
- `GET /api/v1/statistics` - 系统统计

### 使用示例

```python
import requests

# 预测交易
data = {'tx_ids': ['232629023', '230389796']}
response = requests.post(
    'http://localhost:5001/api/v1/predict',
    json=data
)
print(response.json())
```

## 📈 性能指标

### 评估指标

- **AUC-ROC**: 受试者工作特征曲线下面积
- **AUC-PR**: 精确率-召回率曲线下面积
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数

### 评估说明

- val/test 分布不同，阈值需要在 eval 阶段重新校准。
- eval 阶段会输出多阈值对比，便于选择业务阈值。

## 🛠️ 开发指南

### 添加新的图编码器

1. 在`models/dgi.py`中扩展`GINEncoder`类
2. 修改`DGIWithGIN`以支持新编码器
3. 更新配置参数

### 自定义损失函数

1. 在`models/dgi.py`中修改`compute_dgi_loss`方法
2. 实现新的对比学习策略
3. 调整判别器结构

### 扩展特征工程

1. 在`data/feature_engineering.py`中添加新特征
2. 更新`data/__init__.py`中的数据加载流程
3. 调整`num_features`参数

## 🔍 故障排除

### 常见问题

**Q: eval 指标为0？**
```bash
# 说明阈值过高或test分布漂移，需重新校准阈值
python3 run.py --mode eval
```

**Q: API 预测异常全为正常？**
- 确保 API 读取到包含 `calibrated_threshold` 的 eval 文件
- 确保 API 使用全图 Data（已修复）

### 日志分析

训练日志保存在`logs/`目录下：
- `training_*.log` - 训练过程日志
- `model_training.log` - 模型训练日志

```bash
# 查看最新训练日志
tail -f logs/training_$(date +%Y%m%d)_*.log
```

## 📚 参考文献

1. **Deep Graph Infomax** - Velickovic et al., ICLR 2019
2. **How Powerful are Graph Neural Networks?** - Xu et al., ICLR 2019
3. **Elliptic Data Set** - Webb et al., KDD 2019

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者

---

⭐ 如果这个项目对你有帮助，请给它一个星标！
