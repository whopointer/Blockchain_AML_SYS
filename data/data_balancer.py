"""
数据平衡器模块
实现过采样和欠采样策略以处理类别不平衡问题
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.utils import resample
from typing import Tuple, Dict, Any, Optional
import logging


class DataBalancer:
    """
    数据平衡器，提供多种策略处理类别不平衡
    """
    
    def __init__(self, strategy: str = 'undersample', random_state: int = 42):
        """
        初始化数据平衡器
        
        Args:
            strategy: 平衡策略，可选 'undersample', 'oversample', 'hybrid'
            random_state: 随机种子
        """
        self.strategy = strategy
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def balance_graph_data(self, data: Data) -> Data:
        """
        对图数据进行平衡处理
        
        Args:
            data: 原始图数据
            
        Returns:
            平衡后的图数据
        """
        if not hasattr(data, 'y') or data.y is None:
            self.logger.warning("数据没有标签，跳过平衡处理")
            return data
        
        # 获取节点标签
        labels = data.y.numpy()
        balanced_indices = self._get_balanced_indices(labels)
        
        # 创建平衡后的数据
        balanced_data = self._create_balanced_data(data, balanced_indices)
        
        # 验证平衡结果
        balanced_labels = balanced_data.y.numpy()
        balanced_unique, balanced_counts = np.unique(balanced_labels, return_counts=True)
        self.logger.info(f"平衡后数据分布: {dict(zip(balanced_unique, balanced_counts))}")
        
        return balanced_data
    
    def _get_balanced_indices(self, labels: np.ndarray) -> np.ndarray:
        """
        获取平衡后的索引
        
        Args:
            labels: 原始标签
            
        Returns:
            平衡后的索引数组
        """
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        
        self.logger.info(f"原始数据分布: {dict(zip(unique_labels, label_counts))}")
        
        if len(unique_labels) != 2:
            self.logger.warning("只支持二分类数据的平衡处理")
            return np.arange(len(labels))
        
        # 确定多数类和少数类
        majority_class = unique_labels[np.argmax(label_counts)]
        minority_class = unique_labels[np.argmin(label_counts)]
        majority_count = max(label_counts)
        minority_count = min(label_counts)
        
        self.logger.info(f"多数类: {majority_class} (数量: {majority_count})")
        self.logger.info(f"少数类: {minority_class} (数量: {minority_count})")
        
        # 根据策略进行平衡
        if self.strategy == 'undersample':
            balanced_indices = self._undersample(
                labels, majority_class, minority_class, majority_count, minority_count
            )
        elif self.strategy == 'oversample':
            balanced_indices = self._oversample(
                labels, majority_class, minority_class, majority_count, minority_count
            )
        elif self.strategy == 'hybrid':
            balanced_indices = self._hybrid_sampling(
                labels, majority_class, minority_class, majority_count, minority_count
            )
        else:
            raise ValueError(f"未知的平衡策略: {self.strategy}")
        
        return balanced_indices
    
    def _undersample(self, labels: np.ndarray, majority_class: int, minority_class: int,
                    majority_count: int, minority_count: int) -> np.ndarray:
        """欠采样策略：减少多数类样本"""
        majority_indices = np.where(labels == majority_class)[0]
        minority_indices = np.where(labels == minority_class)[0]
        
        # 随机选择与少数类相同数量的多数类样本
        sampled_majority_indices = resample(
            majority_indices,
            n_samples=minority_count,
            replace=False,
            random_state=self.random_state
        )
        
        balanced_indices = np.concatenate([sampled_majority_indices, minority_indices])
        return balanced_indices
    
    def _oversample(self, labels: np.ndarray, majority_class: int, minority_class: int,
                   majority_count: int, minority_count: int) -> np.ndarray:
        """过采样策略：增加少数类样本"""
        majority_indices = np.where(labels == majority_class)[0]
        minority_indices = np.where(labels == minority_class)[0]
        
        # 过采样少数类到与多数类相同数量
        sampled_minority_indices = resample(
            minority_indices,
            n_samples=majority_count,
            replace=True,
            random_state=self.random_state
        )
        
        balanced_indices = np.concatenate([majority_indices, sampled_minority_indices])
        return balanced_indices
    
    def _hybrid_sampling(self, labels: np.ndarray, majority_class: int, minority_class: int,
                        majority_count: int, minority_count: int) -> np.ndarray:
        """混合采样策略：同时对多数类欠采样和对少数类过采样"""
        majority_indices = np.where(labels == majority_class)[0]
        minority_indices = np.where(labels == minority_class)[0]
        
        # 目标数量：取两者之间的中间值
        target_count = (majority_count + minority_count) // 2
        
        # 对多数类欠采样到目标数量
        sampled_majority_indices = resample(
            majority_indices,
            n_samples=min(target_count, len(majority_indices)),
            replace=False,
            random_state=self.random_state
        )
        
        # 对少数类过采样到目标数量
        sampled_minority_indices = resample(
            minority_indices,
            n_samples=target_count,
            replace=True,
            random_state=self.random_state
        )
        
        balanced_indices = np.concatenate([sampled_majority_indices, sampled_minority_indices])
        return balanced_indices
    
    def _create_balanced_data(self, original_data: Data, indices: np.ndarray) -> Data:
        """根据索引创建平衡后的图数据"""
        # 重新索引节点
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
        
        # 创建新的节点特征
        new_x = original_data.x[indices]
        new_y = original_data.y[indices]
        if hasattr(original_data, 'time_steps'):
            new_time_steps = original_data.time_steps[indices]
        
        # 重新构建边索引
        if original_data.edge_index is not None:
            # 筛选只保留平衡后节点之间的边
            mask = torch.isin(original_data.edge_index[0], torch.tensor(indices)) & \
                   torch.isin(original_data.edge_index[1], torch.tensor(indices))
            
            new_edge_index = original_data.edge_index[:, mask]
            
            # 重新映射边索引
            new_edge_index[0] = torch.tensor([old_to_new[idx.item()] for idx in new_edge_index[0]])
            new_edge_index[1] = torch.tensor([old_to_new[idx.item()] for idx in new_edge_index[1]])
        else:
            new_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 创建新的数据对象
        balanced_data = Data(
            x=new_x,
            edge_index=new_edge_index,
            y=new_y
        )
        
        if hasattr(original_data, 'time_steps'):
            balanced_data.time_steps = new_time_steps
        
        return balanced_data


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
