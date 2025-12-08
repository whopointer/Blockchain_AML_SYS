"""
区块链AML数据处理模块
用于加载、预处理和管理Elliptic数据集
"""

import torch
from sympy import false
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# 创建数据模块专用的logger
logger = logging.getLogger(__name__)


class EllipticDataset(Dataset):
    """
    Elliptic区块链数据集类
    处理Elliptic数据集的加载、预处理和图构建
    """
    
    def __init__(self, 
                 root: str, 
                 transform=None, 
                 pre_transform=None,
                 time_step: Optional[int] = None,
                 include_unknown: bool = False,
                 feature_selection: Optional[List[int]] = None):
        """
        初始化数据集
        
        Args:
            root: 数据根目录
            transform: 数据变换
            pre_transform: 预处理变换
            time_step: 特定时间步，None表示使用所有时间步
            include_unknown: 是否包含未知类别样本
            feature_selection: 选择的特征索引列表
        """
        super(EllipticDataset, self).__init__(root, transform, pre_transform)
        self.time_step = time_step
        self.include_unknown = include_unknown
        self.feature_selection = feature_selection
        
        # 加载原始数据
        self._load_raw_data()
        
        # 预处理数据
        self._preprocess_data()
        
        # 构建图数据
        self._build_graph_data()
        
    @property
    def raw_file_names(self) -> List[str]:
        return ['elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv', 'elliptic_txs_features.csv']
    
    @property
    def processed_file_names(self) -> List[str]:
        if self.time_step is not None:
            return [f'graph_data_timestep_{self.time_step}.pt']
        else:
            return ['graph_data_all_timesteps.pt']
    
    def _load_raw_data(self):
        """加载原始数据文件"""
        logger = logging.getLogger(__name__)
        logger.info("开始加载原始数据...")
        
        # 加载类别数据
        self.classes_df = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_classes.csv'))
        
        # 加载边列表
        self.edges_df = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_edgelist.csv'))
        
        # 加载特征数据
        self.features_df = pd.read_csv(
            os.path.join(self.raw_dir, 'elliptic_txs_features.csv'), 
            header=None
        )
        
        # 设置特征列名
        self._set_feature_columns()
        
        logger.info(f"加载完成: {len(self.classes_df)} 个交易, {len(self.edges_df)} 条边")
    
    def _set_feature_columns(self):
        """设置特征列名"""
        # 第一列是txId，第二列是时间步，其余是特征
        self.features_df.columns = ['txId', 'time_step'] + [f'feature_{i}' for i in range(165)]
        
        # 合并特征和类别数据
        self.merged_df = pd.merge(self.features_df, self.classes_df, on='txId', how='left')
    
    def _preprocess_data(self):
        """预处理数据"""
        logger = logging.getLogger(__name__)
        logger.info("开始数据预处理...")
        
        # 过滤数据
        if self.time_step is not None:
            self.merged_df = self.merged_df[self.merged_df['time_step'] == self.time_step]
        
        if not self.include_unknown:
            self.merged_df = self.merged_df[self.merged_df['class'] != 'unknown']
        
        # 特征选择
        if self.feature_selection is not None:
            feature_cols = [f'feature_{i}' for i in self.feature_selection]
            self.merged_df = self.merged_df[['txId', 'time_step', 'class'] + feature_cols]
        
        # 处理缺失值
        self.merged_df = self.merged_df.fillna(0)
        
        # 编码类别标签
        if 'class' in self.merged_df.columns:
            le = LabelEncoder()
            self.merged_df['class_encoded'] = le.fit_transform(
                self.merged_df['class'].astype(str)
            )
            self.label_encoder = le
        else:
            self.merged_df['class_encoded'] = 0
            self.label_encoder = None
        
        logger.info(f"预处理完成: {len(self.merged_df)} 个有效交易")
    
    def _build_graph_data(self):
        """构建图数据"""
        logger = logging.getLogger(__name__)
        logger.info("开始构建图数据...")
        
        # 创建节点映射
        self.node_mapping = {tx_id: idx for idx, tx_id in enumerate(self.merged_df['txId'])}
        self.reverse_mapping = {idx: tx_id for tx_id, idx in self.node_mapping.items()}
        
        # 过滤边数据
        valid_edges = self.edges_df[
            self.edges_df['txId1'].isin(self.node_mapping) & 
            self.edges_df['txId2'].isin(self.node_mapping)
        ]
        
        # 创建边索引
        edge_index_1 = valid_edges['txId1'].map(self.node_mapping).values
        edge_index_2 = valid_edges['txId2'].map(self.node_mapping).values
        # 优化：先转换为numpy数组再创建tensor
        edge_array = np.array([edge_index_1, edge_index_2], dtype=np.int64)
        self.edge_index = torch.from_numpy(edge_array)
        
        # 创建节点特征
        feature_cols = [col for col in self.merged_df.columns if col.startswith('feature_')]
        node_features = self.merged_df[feature_cols].values
        
        # 特征标准化
        self.scaler = StandardScaler()
        node_features = self.scaler.fit_transform(node_features)
        self.x = torch.tensor(node_features, dtype=torch.float)
        
        # 创建标签
        if 'class_encoded' in self.merged_df.columns:
            self.y = torch.tensor(self.merged_df['class_encoded'].values, dtype=torch.long)
        else:
            self.y = None
        
        # 创建时间步信息
        self.time_steps = torch.tensor(self.merged_df['time_step'].values, dtype=torch.long)
        
        logger.info(f"图构建完成: {self.x.shape[0]} 个节点, {self.edge_index.shape[1]} 条边")
    
    def len(self) -> int:
        """返回时间步数量"""
        if self.time_step is not None:
            return 1  # 单一时间步
        else:
            return len(self.time_steps.unique())  # 多个时间步
    
    def get(self, idx: int) -> Data:
        """获取图数据"""
        if self.time_step is not None:
            # 单一时间步，返回整个图
            data = Data(
                x=self.x,
                edge_index=self.edge_index,
                y=self.y,
                time_steps=self.time_steps,
                num_nodes=self.x.shape[0]
            )
        else:
            # 多个时间步，返回指定时间步的子图
            target_time_step = self.time_steps.unique()[idx]
            mask = self.time_steps == target_time_step
            
            # 获取该时间步的节点
            node_indices = torch.nonzero(mask).squeeze()
            node_to_new_id = {old_id.item(): new_id for new_id, old_id in enumerate(node_indices)}
            
            # 创建子图的边索引
            edges = self.edge_index
            mask_edges = torch.isin(edges[0], node_indices) & torch.isin(edges[1], node_indices)
            sub_edges = edges[:, mask_edges]
            
            # 重新映射边索引
            sub_edges[0] = torch.tensor([node_to_new_id[edge.item()] for edge in sub_edges[0]])
            sub_edges[1] = torch.tensor([node_to_new_id[edge.item()] for edge in sub_edges[1]])
            
            # 创建子图数据
            sub_x = self.x[mask]
            sub_y = self.y[mask] if self.y is not None else None
            sub_time_steps = self.time_steps[mask]
            
            data = Data(
                x=sub_x,
                edge_index=sub_edges,
                y=sub_y,
                time_steps=sub_time_steps,
                num_nodes=sub_x.shape[0]
            )
        
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def get_node_info(self, node_idx: int) -> Dict[str, Any]:
        """获取节点信息"""
        tx_id = self.reverse_mapping[node_idx]
        node_data = self.merged_df[self.merged_df['txId'] == tx_id].iloc[0]
        
        return {
            'tx_id': tx_id,
            'time_step': node_data['time_step'],
            'class': node_data.get('class', 'unknown'),
            'class_encoded': node_data.get('class_encoded', -1),
            'features': node_data[[col for col in node_data.index if col.startswith('feature_')]].to_dict()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'num_nodes': self.x.shape[0],
            'num_edges': self.edge_index.shape[1],
            'num_features': self.x.shape[1],
            'time_steps': {
                'min': int(self.time_steps.min()),
                'max': int(self.time_steps.max()),
                'unique_count': len(self.time_steps.unique())
            }
        }
        
        if self.y is not None:
            class_counts = torch.bincount(self.y)
            stats['class_distribution'] = {
                str(i): int(count) for i, count in enumerate(class_counts)
            }
        
        return stats


class EllipticDataLoader:
    """
    Elliptic数据加载器
    提供训练、验证、测试数据的加载功能
    """
    
    def __init__(self, 
                 data_path: str,
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 time_split: bool = True):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据路径
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            time_split: 是否按时间步分割
        """
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.time_split = time_split
        
        # 加载完整数据集
        self.full_dataset = EllipticDataset(root=data_path, include_unknown=False)
        
        # 分割数据集
        self._split_datasets()
    
    def _split_datasets(self):
        """分割数据集"""
        logger = logging.getLogger(__name__)
        logger.info("开始分割数据集...")
        
        if self.time_split:
            # 按时间步分割
            self._split_by_time()
        else:
            # 随机分割
            self._split_randomly()
        
        logger.info("数据集分割完成")
    
    def _split_by_time(self):
        """按时间步分割数据集"""
        # 获取所有时间步
        all_time_steps = sorted(self.full_dataset.time_steps.unique())
        
        # 计算分割点 - 确保每个集合都有数据
        n_steps = len(all_time_steps)
        if n_steps < 3:
            # 如果时间步太少，使用简单分割
            train_steps = all_time_steps[:1]
            val_steps = all_time_steps[1:2] if len(all_time_steps) > 1 else []
            test_steps = all_time_steps[2:] if len(all_time_steps) > 2 else []
        else:
            # 正常分割
            test_start = max(1, int(n_steps * (1 - self.test_size)))
            val_start = max(1, int(test_start * (1 - self.val_size / (1 - self.test_size))))
            
            train_steps = all_time_steps[:val_start]
            val_steps = all_time_steps[val_start:test_start]
            test_steps = all_time_steps[test_start:]
        
        train_steps_list = [int(t) for t in train_steps]
        val_steps_list = [int(t) for t in val_steps]
        test_steps_list = [int(t) for t in test_steps]
        
        logger.info(f"时间步分割: 训练({len(train_steps_list)}步)={train_steps_list[:5]}..., "
                   f"验证({len(val_steps_list)}步)={val_steps_list[:5]}..., "
                   f"测试({len(test_steps_list)}步)={test_steps_list[:5]}...")
        
        # 创建子数据集
        self.train_dataset = self._create_subset_by_time(train_steps_list)
        self.val_dataset = self._create_subset_by_time(val_steps_list)
        self.test_dataset = self._create_subset_by_time(test_steps_list)
        
        
    
    def _split_randomly(self):
        """随机分割数据集"""
        # 获取所有节点索引
        all_indices = list(range(len(self.full_dataset.merged_df)))
        
        # 分割
        train_val_indices, test_indices = train_test_split(
            all_indices, test_size=self.test_size, random_state=self.random_state
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=self.val_size/(1-self.test_size), random_state=self.random_state
        )
        
        # 创建子数据集
        self.train_dataset = self._create_subset_by_indices(train_indices)
        self.val_dataset = self._create_subset_by_indices(val_indices)
        self.test_dataset = self._create_subset_by_indices(test_indices)
    
    def _create_subset_by_time(self, time_steps: List[int]) -> EllipticDataset:
        """根据时间步创建子数据集"""
        subset_dataset = EllipticDataset(
            root=self.data_path,
            include_unknown=False
        )
        
        # 过滤数据
        mask = subset_dataset.merged_df['time_step'].isin(time_steps)
        subset_dataset.merged_df = subset_dataset.merged_df[mask]
        
        # 检查是否有数据
        if len(subset_dataset.merged_df) == 0:
            logger.warning(f"时间步 {time_steps} 没有数据，创建空数据集")
            # 创建空的图数据
            subset_dataset.x = torch.empty((0, 165), dtype=torch.float)  # 假设165个特征
            subset_dataset.edge_index = torch.empty((2, 0), dtype=torch.long)
            subset_dataset.y = torch.empty(0, dtype=torch.long)
            subset_dataset.time_steps = torch.empty(0, dtype=torch.long)
            subset_dataset.scaler = StandardScaler()
            # 使用虚拟数据拟合scaler
            dummy_data = np.random.randn(1, 165)
            subset_dataset.scaler.fit(dummy_data)
        else:
            # 重新构建图
            subset_dataset._build_graph_data()
        
        return subset_dataset
    
    def _create_subset_by_indices(self, indices: List[int]) -> EllipticDataset:
        """根据索引创建子数据集"""
        subset_dataset = EllipticDataset(
            root=self.data_path,
            include_unknown=False
        )
        
        # 过滤数据
        subset_dataset.merged_df = subset_dataset.merged_df.iloc[indices].reset_index(drop=True)
        
        # 重新构建图
        subset_dataset._build_graph_data()
        
        return subset_dataset
    
    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )
    
    def get_val_loader(self, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        """获取验证数据加载器"""
        return DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )
    
    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        return {
            'full_dataset': self.full_dataset.get_statistics(),
            'train_dataset': self.train_dataset.get_statistics(),
            'val_dataset': self.val_dataset.get_statistics(),
            'test_dataset': self.test_dataset.get_statistics()
        }


# 向后兼容的函数
def load_train_data(data_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """加载训练数据（向后兼容）"""
    loader = EllipticDataLoader(data_path)
    return loader.get_train_loader(batch_size, num_workers=num_workers)


def load_val_data(data_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """加载验证数据（向后兼容）"""
    loader = EllipticDataLoader(data_path)
    return loader.get_val_loader(batch_size, num_workers=num_workers)


def load_test_data(data_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    """加载测试数据（向后兼容）"""
    loader = EllipticDataLoader(data_path)
    return loader.get_test_loader(batch_size, num_workers=num_workers)


def load_inference_data(data_path: str, include_unknown: bool = False, 
                       all_timesteps: bool = True, timestep: Optional[int] = None) -> Data:
    """加载推理数据
    
    Args:
        data_path: 数据路径
        include_unknown: 是否包含未知标签的数据，默认为False以避免预测偏差
        all_timesteps: 是否返回所有时间步的完整数据，默认为True
        timestep: 指定返回特定时间步的数据（当all_timesteps=False时使用）
    """
    dataset = EllipticDataset(root=data_path, include_unknown=include_unknown)
    
    if all_timesteps:
        # 返回包含所有时间步的完整图数据
        data = Data(
            x=dataset.x,
            edge_index=dataset.edge_index,
            y=dataset.y,
            time_steps=dataset.time_steps,
            num_nodes=dataset.x.shape[0]
        )
        return data
    elif timestep is not None:
        # 返回指定时间步的数据
        if timestep < 0 or timestep >= len(dataset):
            raise ValueError(f"时间步 {timestep} 超出范围，有效范围: 0-{len(dataset)-1}")
        return dataset[timestep]
    else:
        # 向后兼容：默认返回第一个时间步
        return dataset[0]


def create_sample_data(data_path: str):
    """创建示例数据（向后兼容）"""
    logger = logging.getLogger(__name__)
    logger.warning("Elliptic数据集已存在，无需创建示例数据")


def preprocess_blockchain_data(transactions_path: str, addresses_path: str, 
                              edges_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """预处理区块链数据（向后兼容）"""
    logger = logging.getLogger(__name__)
    logger.warning("请使用EllipticDataset类进行数据处理")


def create_elliptic_dataloader(data_path: str, **kwargs) -> EllipticDataLoader:
    """工厂函数：创建Elliptic数据加载器"""
    return EllipticDataLoader(data_path, **kwargs)


def create_elliptic_dataset(data_path: str, **kwargs) -> EllipticDataset:
    """工厂函数：创建Elliptic数据集"""
    return EllipticDataset(root=data_path, **kwargs)


__all__ = [
    'EllipticDataset',
    'EllipticDataLoader',
    'load_train_data',
    'load_val_data', 
    'load_test_data',
    'load_inference_data',
    'create_sample_data',
    'preprocess_blockchain_data',
    'create_elliptic_dataloader',
    'create_elliptic_dataset'
]