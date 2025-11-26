import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, BatchNorm
from typing import List, Optional, Dict, Any
import numpy as np


class GINLayer(nn.Module):
    """
    改进的GIN层：支持批归一化和dropout
    """
    def __init__(self, in_channels: int, out_channels: int, eps: float = 0, 
                 train_eps: bool = True, dropout: float = 0.1):
        super(GINLayer, self).__init__()
        self.conv = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                BatchNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels, out_channels),
                BatchNorm(out_channels)
            ), eps=eps, train_eps=train_eps)
        
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = in_channels == out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index)
        out = self.dropout(out)
        
        # 残差连接
        if self.residual_connection:
            out = out + x
            
        return out


class MultiScaleGNN(nn.Module):
    """
    多尺度图神经网络层
    """
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 3):
        super(MultiScaleGNN, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            GINLayer(in_channels, out_channels // num_heads, dropout=0.1)
            for _ in range(num_heads)
        ])
        self.fusion = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x, edge_index))
        
        # 拼接多头输出
        out = torch.cat(head_outputs, dim=-1)
        out = self.fusion(out)
        return out


class AttentionPooling(nn.Module):
    """
    注意力池化层
    """
    def __init__(self, in_channels: int):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is not None:
            # 批处理图
            attention_weights = self.attention(x)
            attention_weights = torch.softmax(attention_weights, dim=0)
            return torch.sum(x * attention_weights, dim=0)
        else:
            # 单图
            attention_weights = self.attention(x)
            attention_weights = torch.softmax(attention_weights, dim=0)
            return torch.sum(x * attention_weights, dim=0)


class ImprovedGNNModel(nn.Module):
    """
    改进的GNN模型：用于区块链AML检测
    """
    def __init__(self, 
                 num_features: int, 
                 num_classes: int,
                 hidden_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_multi_scale: bool = False,
                 use_attention_pooling: bool = False):
        super(ImprovedGNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.use_multi_scale = use_multi_scale
        self.use_attention_pooling = use_attention_pooling
        
        # 输入嵌入层
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            BatchNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_channels
            out_channels = hidden_channels
            
            if use_multi_scale and i > 0:
                layer = MultiScaleGNN(in_channels, out_channels)
            else:
                layer = GINLayer(in_channels, out_channels, dropout=dropout)
            
            self.gnn_layers.append(layer)
        
        # 池化层
        if use_attention_pooling:
            self.pooling = AttentionPooling(hidden_channels)
        else:
            self.pooling = nn.Identity()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            BatchNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # 图级别分类器（用于DGI）
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 通过GNN层
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
        
        # 池化
        if batch is not None:
            # 批处理图：全局池化
            pooled = global_mean_pool(x, batch)
        else:
            # 单图：注意力池化或平均池化
            if self.use_attention_pooling:
                pooled = self.pooling(x)
            else:
                pooled = torch.mean(x, dim=0, keepdim=True)
        
        # 分类
        out = self.classifier(pooled)
        return out
    
    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取节点嵌入向量
        """
        x = self.input_embedding(x)
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
        return x
    
    def get_graph_embedding(self, x: torch.Tensor, edge_index: torch.Tensor, 
                           batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取图嵌入向量（用于DGI）
        """
        node_embeddings = self.get_node_embeddings(x, edge_index)
        
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            graph_embedding = torch.mean(node_embeddings, dim=0, keepdim=True)
        
        return self.graph_classifier(graph_embedding)


# 向后兼容的别名
GNNModel = ImprovedGNNModel


def create_model(model_type: str = 'improved', **kwargs) -> nn.Module:
    """
    工厂函数：创建不同类型的模型
    """
    if model_type == 'improved':
        return ImprovedGNNModel(**kwargs)
    elif model_type == 'basic':
        # 基础版本（向后兼容）
        return GNNModel(
            num_features=kwargs.get('num_features', 128),
            num_classes=kwargs.get('num_classes', 2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")