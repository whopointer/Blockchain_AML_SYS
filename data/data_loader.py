"""
高级数据加载器模块
提供灵活的数据加载、批处理和采样策略
"""

import torch
from torch_geometric.data import Data, Dataset, DataLoader, Batch
from torch.utils.data import Sampler, WeightedRandomSampler
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import logging
from collections import defaultdict
import random
from sklearn.utils.class_weight import compute_class_weight
import warnings


class GraphBatchSampler(Sampler):
    """
    图批采样器，支持按类别或时间步采样
    """
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 balance_classes: bool = False,
                 stratify_by: str = 'class'):
        """
        初始化批采样器
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.balance_classes = balance_classes
        self.stratify_by = stratify_by
        
        # 获取数据索引和标签
        self.indices = list(range(len(dataset)))
        self._prepare_stratification()
        
    def _prepare_stratification(self):
        """准备分层采样"""
        if hasattr(self.dataset, 'merged_df'):
            if self.stratify_by == 'class' and 'class' in self.dataset.merged_df.columns:
                self.strata = defaultdict(list)
                for idx, row in self.dataset.merged_df.iterrows():
                    self.strata[row['class']].append(idx)
            elif self.stratify_by == 'time_step' and 'time_step' in self.dataset.merged_df.columns:
                self.strata = defaultdict(list)
                for idx, row in self.dataset.merged_df.iterrows():
                    self.strata[row['time_step']].append(idx)
            else:
                self.strata = None
        else:
            self.strata = None
    
    def __iter__(self):
        if self.balance_classes and self.strata:
            # 类别平衡采样
            yield from self._balanced_sampling()
        elif self.strata:
            # 分层采样
            yield from self._stratified_sampling()
        else:
            # 随机采样
            indices = self.indices.copy()
            if self.shuffle:
                random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    yield batch_indices
    
    def _balanced_sampling(self):
        """类别平衡采样"""
        min_strata_size = min(len(stratum) for stratum in self.strata.values())
        
        # 每个类别采样的样本数
        samples_per_class = min(self.batch_size // len(self.strata), min_strata_size)
        
        while True:
            batch_indices = []
            for stratum_indices in self.strata.values():
                sampled = random.sample(stratum_indices, samples_per_class)
                batch_indices.extend(sampled)
            
            if len(batch_indices) >= self.batch_size:
                yield batch_indices[:self.batch_size]
            else:
                if not self.drop_last:
                    yield batch_indices
                break
    
    def _stratified_sampling(self):
        """分层采样"""
        # 按比例采样
        total_size = sum(len(stratum) for stratum in self.strata.values())
        
        for i in range(0, total_size, self.batch_size):
            batch_indices = []
            remaining = self.batch_size
            
            for stratum_name, stratum_indices in self.strata.items():
                if remaining <= 0:
                    break
                
                # 计算该类别在批次中应有的样本数
                stratum_ratio = len(stratum_indices) / total_size
                stratum_batch_size = int(remaining * stratum_ratio)
                
                # 采样
                if len(stratum_indices) >= stratum_batch_size:
                    sampled = random.sample(stratum_indices, stratum_batch_size)
                    batch_indices.extend(sampled)
                    remaining -= stratum_batch_size
            
            if len(batch_indices) >= self.batch_size or not self.drop_last:
                yield batch_indices[:self.batch_size]
    
    def __len__(self):
        if self.strata and self.balance_classes:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size


class AdvancedDataLoader:
    """
    高级数据加载器，整合多种加载策略
    """
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = False):
        """
        初始化高级数据加载器
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # 计算类别权重
        self._compute_class_weights()
    
    def _compute_class_weights(self):
        """计算类别权重"""
        if hasattr(self.dataset, 'y') and self.dataset.y is not None:
            try:
                labels = self.dataset.y.numpy()
                self.class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(labels),
                    y=labels
                )
                self.class_weights = torch.FloatTensor(self.class_weights)
            except Exception as e:
                logging.warning(f"计算类别权重失败: {e}")
                self.class_weights = None
        else:
            self.class_weights = None
    
    def get_balanced_loader(self, shuffle: bool = True) -> DataLoader:
        """获取类别平衡的数据加载器"""
        if self.class_weights is not None:
            # 为每个样本分配权重
            sample_weights = self.class_weights[self.dataset.y]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False
            )
        else:
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers if self.num_workers > 0 else False
            )
    
    def get_standard_loader(self, shuffle: bool = True) -> DataLoader:
        """获取标准数据加载器"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False
        )


def create_dataloader(dataset: Dataset,
                     loader_type: str = 'standard',
                     **kwargs) -> DataLoader:
    """
    工厂函数：创建数据加载器
    """
    advanced_loader = AdvancedDataLoader(dataset, **kwargs)
    
    if loader_type == 'standard':
        return advanced_loader.get_standard_loader()
    elif loader_type == 'balanced':
        return advanced_loader.get_balanced_loader()
    else:
        raise ValueError(f"未知的加载器类型: {loader_type}")


__all__ = [
    'GraphBatchSampler',
    'AdvancedDataLoader',
    'create_dataloader'
]