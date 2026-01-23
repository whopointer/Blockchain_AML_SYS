"""
数据平衡器模块
实现过采样和欠采样策略以处理类别不平衡问题
"""


import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch_geometric.data import Data


class DataBalancer:
    """
    数据平衡器（适配：单大图节点分类 / Elliptic / train_mask）

    重要说明：
    - 不重建图结构，不删除节点，不改 edge_index
    - 只对训练节点(train_mask)的“用于计算loss的索引集合”做平衡
    - 输出结果会挂载到 data 上：
        data.balanced_train_idx : torch.LongTensor  (用于算loss的节点索引)
        data.pos_weight         : torch.FloatTensor or None (用于 BCEWithLogitsLoss)
    """

    def __init__(self, strategy: str = "undersample", random_state: int = 42):
        """
        Args:
            strategy:
                'undersample'   : 欠采样多数类（推荐用于监督训练阶段）
                'weighted_loss' : 不采样，返回 pos_weight（最推荐，最稳）
                'none'          : 不处理
                兼容旧参数：
                'oversample'/'hybrid'：在单大图GNN监督阶段不建议，会自动降级为 weighted_loss
            random_state: 随机种子
        """
        self.strategy = strategy
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def balance_graph_data(self, data: Data, ignore_label: Optional[int] = None) -> Data:
        """
        对图数据进行“训练节点层面”的平衡处理（不改变图本体）

        Args:
            data: 原始图数据（单大图），需要 data.y 和 data.train_mask
            ignore_label: 若 y 中存在“未知/不参与监督”的标签值（例如 2 或 -1），在平衡与loss计算时剔除

        Returns:
            原 data（图不变），但会额外挂载：
                data.balanced_train_idx
                data.pos_weight
        """
        if not hasattr(data, "y") or data.y is None:
            self.logger.warning("数据没有标签 y，跳过平衡处理")
            return data

        if not hasattr(data, "train_mask") or data.train_mask is None:
            self.logger.warning("数据没有 train_mask，单大图节点分类建议提供 train_mask；跳过平衡处理")
            return data

        y = data.y
        train_mask = data.train_mask

        # 只取训练节点索引
        train_idx = torch.nonzero(train_mask, as_tuple=False).view(-1)
        if train_idx.numel() == 0:
            self.logger.warning("train_mask 为空，跳过平衡处理")
            data.balanced_train_idx = train_idx
            data.pos_weight = None
            return data

        # 过滤 ignore_label（例如 unknown）
        y_flat = y.view(-1)
        if ignore_label is not None:
            valid = y_flat[train_idx] != ignore_label
            train_idx = train_idx[valid]

        if train_idx.numel() == 0:
            self.logger.warning("过滤 ignore_label 后训练节点为空，跳过平衡处理")
            data.balanced_train_idx = train_idx
            data.pos_weight = None
            return data

        # 统计训练集分布（只统计 0/1）
        y_train = y_flat[train_idx]
        # 如果不是严格0/1，仍然统计看看
        unique, counts = torch.unique(y_train, return_counts=True)
        dist = {int(k.item()): int(v.item()) for k, v in zip(unique, counts)}
        self.logger.info(f"训练集原始分布(基于 train_mask): {dist}")

        # 兼容旧策略名，但在单大图GNN监督阶段不做 oversample/hybrid
        if self.strategy in ("oversample", "hybrid"):
            self.logger.warning(
                f"strategy='{self.strategy}' 在单大图GNN监督阶段不建议（节点克隆会破坏边映射），"
                f"已自动降级为 'weighted_loss'"
            )
            effective_strategy = "weighted_loss"
        else:
            effective_strategy = self.strategy

        # 默认输出
        balanced_train_idx = train_idx
        pos_weight = None

        if effective_strategy == "none":
            pass

        elif effective_strategy == "weighted_loss":
            # pos_weight = N_neg / N_pos (用于 BCEWithLogitsLoss)
            count0 = int((y_train == 0).sum().item())
            count1 = int((y_train == 1).sum().item())

            if count0 == 0 or count1 == 0:
                self.logger.warning("训练集中某一类数量为0，pos_weight 无意义；将不设置 pos_weight")
                pos_weight = None
            else:
                pos_weight = torch.tensor([count0 / count1], dtype=torch.float, device=y.device)

        elif effective_strategy == "undersample":
            balanced_train_idx = self._undersample_train_indices(train_idx, y_flat, device=y.device)

        else:
            raise ValueError(f"未知的平衡策略: {self.strategy}")

        # 挂载到 data 上，供训练阶段直接使用
        data.balanced_train_idx = balanced_train_idx
        data.pos_weight = pos_weight

        # 打印平衡后分布
        if balanced_train_idx.numel() > 0:
            y_bal = y_flat[balanced_train_idx]
            u2, c2 = torch.unique(y_bal, return_counts=True)
            dist2 = {int(k.item()): int(v.item()) for k, v in zip(u2, c2)}
            self.logger.info(f"训练集平衡后分布(用于loss的索引集合): {dist2}")

        if pos_weight is not None:
            self.logger.info(f"pos_weight (for BCEWithLogitsLoss): {pos_weight.detach().cpu().numpy().tolist()}")

        return data

    def _undersample_train_indices(self, train_idx: torch.Tensor, y_flat: torch.Tensor,
                                   device: torch.device) -> torch.Tensor:
        """
        欠采样：只在 train_idx 里做多数类下采样，返回平衡后的 train_idx 子集
        """
        y_train = y_flat[train_idx]

        idx0 = train_idx[y_train == 0]
        idx1 = train_idx[y_train == 1]


def create_balanced_loader(data: Data, batch_size: int = 64,
                          balance_strategy: str = 'undersample',
                          shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    创建平衡后的数据加载器
    
    Args:
        data: 原始图数据
        batch_size: 批次大小
        balance_strategy: 平衡策略
        shuffle: 是否打乱数据
        
    Returns:
        平衡后的数据加载器
    """
    from torch_geometric.loader import DataLoader
    
    # 应用数据平衡
    balancer = DataBalancer(strategy=balance_strategy)
    balanced_data = balancer.balance_graph_data(data)
    
    # 创建数据加载器
    loader = DataLoader(
        [balanced_data],  # 将单个图数据包装成列表
        batch_size=1,     # 对于图数据，通常使用批次大小1
        shuffle=shuffle
    )
    
    return loader


__all__ = ['DataBalancer', 'create_balanced_loader']
