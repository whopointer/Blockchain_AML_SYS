"""
模型模块
包含图神经网络模型、训练器、推理引擎和评估器
"""

from .gnn_model import (
    GINLayer,
    MultiScaleGNN,
    AttentionPooling,
    ImprovedGNNModel,
    GNNModel,
    create_model
)

from .dgi import (
    ImprovedDGI,
    ContrastiveDGI,
    DGI,
    create_dgi_model
)

from .trainer import (
    MetricsTracker,
    ImprovedTrainer,
    train,
    evaluate,
    create_trainer
)

from .inference import (
    InferenceEngine,
    inference,
    create_inference_engine
)

from .evaluator import (
    ModelEvaluator,
    evaluate,
    create_evaluator
)

__all__ = [
    # GNN模型
    'GINLayer',
    'MultiScaleGNN',
    'AttentionPooling',
    'ImprovedGNNModel',
    'GNNModel',
    'create_model',
    
    # DGI模型
    'ImprovedDGI',
    'ContrastiveDGI',
    'DGI',
    'create_dgi_model',
    
    # 训练器
    'MetricsTracker',
    'ImprovedTrainer',
    'train',
    'evaluate',
    'create_trainer',
    
    # 推理引擎
    'InferenceEngine',
    'inference',
    'create_inference_engine',
    
    # 评估器
    'ModelEvaluator',
    'evaluate',
    'create_evaluator'
]