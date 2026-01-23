"""
Deep Graph Infomax (DGI) 实现
使用GIN编码器的自监督图表示学习
"""




from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool


class GINEncoder(nn.Module):
    """
    GIN Encoder（风格A）：
    - BatchNorm 只放在 GINConv 内部的 MLP 里（nn.BatchNorm1d）
    - conv 输出后：非最后一层做 ReLU + Dropout
    """
    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        def make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            """
            经典 GIN MLP（两层）：
            Linear -> BN -> ReLU -> Linear -> BN
            注意：不在 MLP 末尾再接 ReLU，把“是否激活”交给外层（更常见、更干净）
            """
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim),
            )

        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")

        # 第一层：num_features -> hidden
        self.convs.append(GINConv(make_mlp(num_features, hidden_channels)))

        # 后续层：hidden -> hidden
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels)))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # 非最后一层：ReLU + Dropout（最后一层保留线性输出，更利于下游任务）
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = self.drop(x)

        return x

    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.forward(x, edge_index)


class DGIWithGIN(nn.Module):
    """
    DGI + GIN：
    - 第一阶段：DGI 自监督训练 encoder（gin_encoder）
    - 第二阶段：freeze encoder，拿 embeddings 给下游分类器/传统模型（比如 RF）
    """

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        pooling_strategy: str = "mean",
        corruption_method: str = "feature_shuffle",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.pooling_strategy = pooling_strategy
        self.corruption_method = corruption_method

        # GIN 编码器
        self.gin_encoder = GINEncoder(
            num_features=num_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

        # 判别器：双线性打分，输出 logits（不要手动 sigmoid）
        self.discriminator = nn.Bilinear(hidden_channels, hidden_channels, 1)

        # 摘要函数：对图级 embedding 再做一次 MLP（增强表达）
        self.summary_function = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.is_pretrained = False

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
        - positive_scores: [num_nodes]  每个节点与其所属图摘要的匹配分数（logits）
        - negative_scores: [num_nodes]
        """

        # 正样本节点嵌入
        positive_embeddings = self.gin_encoder.get_node_embeddings(x, edge_index)

        # 负样本：特征扰动
        corrupted_x = self._corrupt_features(x)
        negative_embeddings = self.gin_encoder.get_node_embeddings(corrupted_x, edge_index)

        # batch 为空则当作单图
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 图级摘要：pooling 得到 [num_graphs, hidden]
        graph_summary = self._pool_embeddings(positive_embeddings, batch)

        # 摘要增强 + sigmoid（更贴近经典 DGI，训练更稳）
        graph_summary = torch.sigmoid(self.summary_function(graph_summary))

        # 关键修复：每个节点取它所属图的摘要向量，对齐到 [num_nodes, hidden]
        summary_per_node = graph_summary[batch]

        # 判别器打分（输出 logits）
        positive_scores = self.discriminator(positive_embeddings, summary_per_node).squeeze(-1)
        negative_scores = self.discriminator(negative_embeddings, summary_per_node).squeeze(-1)

        return positive_scores, negative_scores

    def _corrupt_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        负样本生成：特征扰动（默认 feature_shuffle）
        """
        if self.corruption_method == "feature_shuffle":
            # 关键修复：randperm 必须在同一 device 上，否则 CUDA 会报错
            perm = torch.randperm(x.size(0), device=x.device)
            return x[perm]

        elif self.corruption_method == "feature_noise":
            noise = torch.randn_like(x) * 0.1
            return x + noise

        elif self.corruption_method == "feature_mask":
            # 随机掩码：80% 保留
            mask = (torch.rand_like(x) > 0.2).float()
            return x * mask

        else:
            return x

    def _pool_embeddings(self, embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [num_nodes, hidden]
        batch:      [num_nodes] 每个节点所属图 id
        return:     [num_graphs, hidden]
        """
        if self.pooling_strategy == "mean":
            return global_mean_pool(embeddings, batch)
        elif self.pooling_strategy == "max":
            return global_max_pool(embeddings, batch)
        else:
            return global_mean_pool(embeddings, batch)

    def compute_dgi_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """
        DGI loss：BCEWithLogits
        - 正样本标签 1
        - 负样本标签 0
        """
        pos_labels = torch.ones_like(positive_scores)
        neg_labels = torch.zeros_like(negative_scores)

        pos_loss = F.binary_cross_entropy_with_logits(positive_scores, pos_labels)
        neg_loss = F.binary_cross_entropy_with_logits(negative_scores, neg_labels)

        return pos_loss + neg_loss

    @torch.no_grad()
    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        第二阶段用：输出节点嵌入
        """
        self.eval()
        return self.gin_encoder.get_node_embeddings(x, edge_index)

    def freeze_encoder(self) -> None:
        """
        第二阶段：冻结 encoder（只训练下游分类器）
        """
        for p in self.gin_encoder.parameters():
            p.requires_grad = False
        self.is_pretrained = True

    def unfreeze_encoder(self) -> None:
        """
        解冻 encoder
        """
        for p in self.gin_encoder.parameters():
            p.requires_grad = True
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