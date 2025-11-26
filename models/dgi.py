import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import negative_sampling
from typing import Optional, Tuple, Dict, Any
import numpy as np


class ImprovedDGI(nn.Module):
    """
    改进的Deep Graph Infomax (DGI) 用于自监督学习节点嵌入
    支持多种池化策略和负采样方法
    """

    def __init__(self, 
                 gnn_model: nn.Module, 
                 hidden_channels: int,
                 pooling_strategy: str = 'mean',
                 corruption_method: str = 'shuffle',
                 temperature: float = 0.5):
        super(ImprovedDGI, self).__init__()
        self.gnn_model = gnn_model
        self.hidden_channels = hidden_channels
        self.pooling_strategy = pooling_strategy
        self.corruption_method = corruption_method
        self.temperature = temperature
        
        # 判别器网络
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        # 摘要函数
        self.summary_function = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        """
        # 获取节点嵌入
        node_embeddings = self.gnn_model.get_node_embeddings(x, edge_index)
        
        # 计算图级别摘要
        if batch is not None:
            # 批处理图
            graph_summary = self._pool_graph_embeddings(node_embeddings, batch)
        else:
            # 单图
            graph_summary = self._pool_graph_embeddings(
                node_embeddings, 
                torch.zeros(node_embeddings.size(0), dtype=torch.long, device=x.device)
            )
        
        # 应用摘要函数
        graph_summary = self.summary_function(graph_summary)
        
        # 计算正样本分数
        positive_scores = self._compute_discriminator_scores(node_embeddings, graph_summary)
        
        return positive_scores

    def _pool_graph_embeddings(self, embeddings: torch.Tensor, 
                               batch: torch.Tensor) -> torch.Tensor:
        """
        图嵌入池化
        """
        if self.pooling_strategy == 'mean':
            return global_mean_pool(embeddings, batch)
        elif self.pooling_strategy == 'max':
            return global_max_pool(embeddings, batch)
        elif self.pooling_strategy == 'add':
            return global_add_pool(embeddings, batch)
        elif self.pooling_strategy == 'attention':
            return self._attention_pooling(embeddings, batch)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def _attention_pooling(self, embeddings: torch.Tensor, 
                           batch: torch.Tensor) -> torch.Tensor:
        """
        注意力池化
        """
        batch_size = batch.max().item() + 1
        pooled_embeddings = []
        
        for i in range(batch_size):
            mask = (batch == i)
            node_emb = embeddings[mask]
            
            # 简单的注意力机制
            attention_weights = torch.mean(node_emb, dim=1, keepdim=True)
            attention_weights = F.softmax(attention_weights / self.temperature, dim=0)
            pooled_emb = torch.sum(node_emb * attention_weights, dim=0, keepdim=True)
            pooled_embeddings.append(pooled_emb)
        
        return torch.cat(pooled_embeddings, dim=0)

    def _compute_discriminator_scores(self, node_embeddings: torch.Tensor, 
                                      graph_summary: torch.Tensor) -> torch.Tensor:
        """
        计算判别器分数
        """
        # 扩展图摘要到与节点嵌入相同的大小
        if graph_summary.size(0) == 1:
            graph_summary_expanded = graph_summary.expand(node_embeddings.size(0), -1)
        else:
            graph_summary_expanded = graph_summary
        
        # 计算分数
        scores = self.discriminator(node_embeddings * graph_summary_expanded)
        return scores.squeeze(-1)

    def corruption(self, edge_index: torch.Tensor, num_nodes: int, 
                   batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成负样本
        """
        if self.corruption_method == 'shuffle':
            # 边洗牌
            return self._shuffle_edges(edge_index)
        elif self.corruption_method == 'negative_sampling':
            # 负采样
            return negative_sampling(edge_index, num_nodes=num_nodes)
        elif self.corruption_method == 'feature_corruption':
            # 特征损坏（在训练时处理）
            return edge_index
        else:
            raise ValueError(f"Unknown corruption method: {self.corruption_method}")

    def _shuffle_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        边洗牌
        """
        edge_index_shuffled = edge_index[:, torch.randperm(edge_index.size(1))]
        return edge_index_shuffled

    def compute_loss(self, positive_scores: torch.Tensor, 
                    negative_scores: torch.Tensor) -> torch.Tensor:
        """
        计算DGI损失
        """
        # 正样本标签为1，负样本标签为0
        positive_labels = torch.ones_like(positive_scores)
        negative_labels = torch.zeros_like(negative_scores)
        
        # 计算二元交叉熵损失
        pos_loss = F.binary_cross_entropy_with_logits(positive_scores, positive_labels)
        neg_loss = F.binary_cross_entropy_with_logits(negative_scores, negative_labels)
        
        total_loss = pos_loss + neg_loss
        return total_loss

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取学习到的节点嵌入
        """
        return self.gnn_model.get_node_embeddings(x, edge_index)


# 向后兼容的别名
DGI = ImprovedDGI


class ContrastiveDGI(nn.Module):
    """
    对比学习版本的DGI
    """
    def __init__(self, gnn_model: nn.Module, hidden_channels: int):
        super(ContrastiveDGI, self).__init__()
        self.gnn_model = gnn_model
        self.hidden_channels = hidden_channels
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_index_corrupted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回正负样本的投影嵌入
        """
        # 正样本嵌入
        pos_embeddings = self.gnn_model.get_node_embeddings(x, edge_index)
        pos_projections = self.projection_head(pos_embeddings)
        
        # 负样本嵌入
        neg_embeddings = self.gnn_model.get_node_embeddings(x, edge_index_corrupted)
        neg_projections = self.projection_head(neg_embeddings)
        
        return pos_projections, neg_projections

    def compute_contrastive_loss(self, pos_embeddings: torch.Tensor, 
                                neg_embeddings: torch.Tensor, 
                                temperature: float = 0.5) -> torch.Tensor:
        """
        计算对比损失
        """
        # L2归一化
        pos_embeddings = F.normalize(pos_embeddings, dim=1)
        neg_embeddings = F.normalize(neg_embeddings, dim=1)
        
        # 计算相似度矩阵
        batch_size = pos_embeddings.size(0)
        
        # 正样本相似度
        pos_sim = torch.sum(pos_embeddings * pos_embeddings, dim=1) / temperature
        
        # 负样本相似度
        neg_sim_matrix = torch.mm(pos_embeddings, neg_embeddings.t()) / temperature
        
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim_matrix], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=pos_embeddings.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


def create_dgi_model(model_type: str = 'improved', **kwargs) -> nn.Module:
    """
    工厂函数：创建不同类型的DGI模型
    """
    if model_type == 'improved':
        return ImprovedDGI(**kwargs)
    elif model_type == 'contrastive':
        return ContrastiveDGI(**kwargs)
    elif model_type == 'basic':
        # 基础版本（向后兼容）
        return DGI(
            gnn_model=kwargs.get('gnn_model'),
            hidden_channels=kwargs.get('hidden_channels', 64)
        )
    else:
        raise ValueError(f"Unknown DGI model type: {model_type}")