"""
图构建器模块
用于从原始数据构建图结构
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import logging


class GraphBuilder:
    """
    图构建器，用于从交易数据构建图结构
    """
    
    def __init__(self):
        self.graph_cache = {}
        self.node_mappings = {}
    
    def build_transaction_graph(self,
                               transactions: pd.DataFrame,
                               features: pd.DataFrame,
                               node_column: str = 'txId',
                               from_column: str = 'txId1',
                               to_column: str = 'txId2',
                               directed: bool = False) -> Data:
        """
        构建交易图
        """
        logger = logging.getLogger(__name__)
        logger.info("开始构建交易图...")
        
        # 创建节点映射
        all_nodes = set(features[node_column].unique())
        transaction_nodes = set(transactions[from_column].unique()) | set(transactions[to_column].unique())
        
        # 只保留有交易记录的节点
        valid_nodes = all_nodes.intersection(transaction_nodes)
        valid_features = features[features[node_column].isin(valid_nodes)]
        
        # 创建节点ID映射
        node_to_idx = {node_id: idx for idx, node_id in enumerate(valid_features[node_column])}
        
        # 构建边索引
        valid_transactions = transactions[
            transactions[from_column].isin(node_to_idx) & 
            transactions[to_column].isin(node_to_idx)
        ]
        
        edge_index_1 = valid_transactions[from_column].map(node_to_idx).values
        edge_index_2 = valid_transactions[to_column].map(node_to_idx).values
        edge_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
        
        # 构建节点特征
        feature_columns = [col for col in valid_features.columns if col != node_column]
        node_features = valid_features[feature_columns].values
        x = torch.tensor(node_features, dtype=torch.float)
        
        # 创建图数据
        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=len(valid_nodes)
        )
        
        # 如果是无向图，添加反向边
        if not directed:
            data.edge_index = to_undirected(data.edge_index)
        
        # 保存映射关系
        self.node_mappings['transaction'] = node_to_idx
        
        logger.info(f"交易图构建完成: {data.num_nodes} 个节点, {data.edge_index.shape[1]} 条边")
        
        return data
    
    def validate_graph(self, data: Data) -> Dict[str, Any]:
        """验证图数据"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 检查基本属性
        if not hasattr(data, 'x') or data.x is None:
            validation_results['errors'].append("缺少节点特征")
            validation_results['is_valid'] = False
        
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            validation_results['errors'].append("缺少边索引")
            validation_results['is_valid'] = False
        
        if validation_results['is_valid']:
            # 检查维度一致性
            num_nodes = data.x.shape[0]
            max_node_idx = data.edge_index.max().item()
            
            if max_node_idx >= num_nodes:
                validation_results['errors'].append(f"边索引超出节点范围: {max_node_idx} >= {num_nodes}")
                validation_results['is_valid'] = False
        
        return validation_results


# 工厂函数
def create_graph_builder() -> GraphBuilder:
    """创建图构建器"""
    return GraphBuilder()


def build_basic_graph(transactions: pd.DataFrame, features: pd.DataFrame) -> Data:
    """构建基础图"""
    builder = GraphBuilder()
    return builder.build_transaction_graph(transactions, features)


def validate_graph_data(data: Data) -> bool:
    """验证图数据"""
    builder = GraphBuilder()
    results = builder.validate_graph(data)
    return results['is_valid']


__all__ = [
    'GraphBuilder',
    'create_graph_builder',
    'build_basic_graph',
    'validate_graph_data'
]