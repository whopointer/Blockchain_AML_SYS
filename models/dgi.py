"""
Deep Graph Infomax (DGI) 实现
使用GIN编码器的自监督图表示学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import negative_sampling
from typing import Optional, Tuple, Dict, Any
import numpy as np


class GINEncoder(nn.Module):
    """
    GIN编码器，用于DGI中的节点嵌入学习
    具有强大的图同构判别能力
    """
    
    def __init__(self, num_features: int, hidden_channels: int, num_layers: int = 3):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # GIN层
        from torch_geometric.nn import GINConv, BatchNorm
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            )
        ))
        self.norms.append(BatchNorm(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU()
                )
            ))
            self.norms.append(BatchNorm(hidden_channels))
        
        # 最后一层
        if num_layers > 1:
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels)
                )
            ))
            self.norms.append(BatchNorm(hidden_channels))
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            if i < len(self.convs) - 1:  # 最后一层不加ReLU
                x = F.relu(x)
                x = self.dropout(x)
        
        return x
    
    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取节点嵌入
        """
        return self.forward(x, edge_index)


class DGIWithGIN(nn.Module):
    """
    使用GIN编码器的Deep Graph Infomax实现
    实现真正的两阶段训练：自监督学习 + 监督分类
    """
    
    def __init__(self, 
                 num_features: int,
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 pooling_strategy: str = 'mean',
                 corruption_method: str = 'feature_shuffle'):
        super(DGIWithGIN, self).__init__()
        
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.pooling_strategy = pooling_strategy
        self.corruption_method = corruption_method
        
        # GIN编码器
        self.gin_encoder = GINEncoder(
            num_features=num_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )
        
        # 判别器 - 简化为双线性变换
        self.discriminator = nn.Bilinear(hidden_channels, hidden_channels, 1)
        
        # 摘要函数（用于生成图级别表示）
        self.summary_function = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 训练状态标记
        self.is_pretrained = False
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回正样本分数和负样本分数
        """
        # 获取节点嵌入（正样本）
        positive_embeddings = self.gin_encoder.get_node_embeddings(x, edge_index)
        
        # 生成负样本（特征扰动）
        corrupted_x = self._corrupt_features(x)
        negative_embeddings = self.gin_encoder.get_node_embeddings(corrupted_x, edge_index)
        
        # 计算图级别摘要
        if batch is not None:
            graph_summary = self._pool_embeddings(positive_embeddings, batch)
        else:
            # 单图处理
            batch_tensor = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_summary = self._pool_embeddings(positive_embeddings, batch_tensor)
        
        # 应用摘要函数
        graph_summary = self.summary_function(graph_summary)
        
        # 计算判别器分数
        positive_scores = self._compute_discriminator_scores(positive_embeddings, graph_summary)
        negative_scores = self._compute_discriminator_scores(negative_embeddings, graph_summary)
        
        return positive_scores, negative_scores
    
    def _corrupt_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征扰动生成负样本
        """
        if self.corruption_method == 'feature_shuffle':
            # 随机打乱特征
            return x[torch.randperm(x.size(0))]
        elif self.corruption_method == 'feature_noise':
            # 添加高斯噪声
            noise = torch.randn_like(x) * 0.1
            return x + noise
        elif self.corruption_method == 'feature_mask':
            # 随机掩码部分特征
            mask = torch.rand_like(x) > 0.2  # 80%的特征保留
            return x * mask.float()
        else:
            return x
    
    def _pool_embeddings(self, embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        图嵌入池化
        """
        if self.pooling_strategy == 'mean':
            return global_mean_pool(embeddings, batch)
        elif self.pooling_strategy == 'max':
            return global_max_pool(embeddings, batch)
        else:
            return global_mean_pool(embeddings, batch)
    
    def _compute_discriminator_scores(self, node_embeddings: torch.Tensor, 
                                    graph_summary: torch.Tensor) -> torch.Tensor:
        """
        计算判别器分数
        """
        # 使用双线性判别器计算节点嵌入与图摘要的匹配分数
        if graph_summary.size(0) == 1:
            # 单图情况：扩展图摘要到所有节点
            graph_summary_expanded = graph_summary.expand(node_embeddings.size(0), -1)
            scores = self.discriminator(node_embeddings, graph_summary_expanded)
        else:
            # 批处理情况：对应计算
            batch_size = graph_summary.size(0)
            num_nodes_per_graph = node_embeddings.size(0) // batch_size
            scores_list = []
            
            for i in range(batch_size):
                start_idx = i * num_nodes_per_graph
                end_idx = start_idx + num_nodes_per_graph
                batch_embeddings = node_embeddings[start_idx:end_idx]
                batch_summary = graph_summary[i:i+1].expand(num_nodes_per_graph, -1)
                
                batch_scores = self.discriminator(batch_embeddings, batch_summary)
                scores_list.append(batch_scores)
            
            scores = torch.cat(scores_list, dim=0)
        
        return scores.squeeze(-1)
    
    def compute_dgi_loss(self, positive_scores: torch.Tensor, 
                        negative_scores: torch.Tensor) -> torch.Tensor:
        """
        计算DGI损失
        """
        # 正样本标签为1，负样本标签为0
        positive_labels = torch.ones_like(positive_scores)
        negative_labels = torch.zeros_like(negative_scores)
        
        # 二元交叉熵损失
        pos_loss = F.binary_cross_entropy_with_logits(positive_scores, positive_labels)
        neg_loss = F.binary_cross_entropy_with_logits(negative_scores, negative_labels)
        
        return pos_loss + neg_loss
    
    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        获取学习到的节点嵌入（用于第二阶段）
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.gin_encoder.get_node_embeddings(x, edge_index)
        return embeddings
    
    def freeze_encoder(self):
        """
        冻结GIN编码器参数（用于第二阶段训练）
        """
        for param in self.gin_encoder.parameters():
            param.requires_grad = False
        self.is_pretrained = True
    
    def unfreeze_encoder(self):
        """
        解冻GIN编码器参数
        """
        for param in self.gin_encoder.parameters():
            param.requires_grad = True
        self.is_pretrained = False


def create_dgi_with_gin(num_features: int, hidden_channels: int = 128, 
                       num_layers: int = 3, **kwargs) -> DGIWithGIN:
    """
    工厂函数：创建DGI with GIN模型
    """
    return DGIWithGIN(
        num_features=num_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        **kwargs
    )


# 向后兼容
DGI = DGIWithGIN