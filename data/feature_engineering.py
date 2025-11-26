"""
特征工程模块
为区块链AML检测创建高级特征
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import networkx as nx
from torch_geometric.utils import to_networkx
from collections import defaultdict, Counter
import warnings


class GraphFeatureExtractor:
    """
    图特征提取器，从图结构中提取特征
    """
    
    def __init__(self):
        self.graph_features = {}
    
    def extract_node_centrality_features(self, data) -> pd.DataFrame:
        """提取节点中心性特征"""
        # 转换为NetworkX图
        G = to_networkx(data, to_undirected=True)
        
        features = pd.DataFrame(index=range(data.num_nodes))
        
        # 度中心性
        degree_centrality = nx.degree_centrality(G)
        features['degree_centrality'] = [degree_centrality.get(i, 0) for i in range(data.num_nodes)]
        
        # 聚类系数
        clustering = nx.clustering(G)
        features['clustering_coefficient'] = [clustering.get(i, 0) for i in range(data.num_nodes)]
        
        # PageRank
        pagerank = nx.pagerank(G)
        features['pagerank'] = [pagerank.get(i, 0) for i in range(data.num_nodes)]
        
        self.graph_features['centrality'] = features
        return features
    
    def extract_neighborhood_features(self, data) -> pd.DataFrame:
        """提取邻域特征"""
        G = to_networkx(data, to_undirected=True)
        
        features = pd.DataFrame(index=range(data.num_nodes))
        
        # 基本邻域统计
        degrees = dict(G.degree())
        features['degree'] = [degrees.get(i, 0) for i in range(data.num_nodes)]
        
        # 三角形数量
        triangles = nx.triangles(G)
        features['triangles'] = [triangles.get(i, 0) for i in range(data.num_nodes)]
        
        self.graph_features['neighborhood'] = features
        return features


class TransactionFeatureExtractor:
    """
    交易特征提取器，从交易模式中提取特征
    """
    
    def __init__(self):
        self.transaction_features = {}
    
    def extract_amount_features(self, 
                               transactions: pd.DataFrame,
                               amount_column: str = 'amount') -> pd.DataFrame:
        """提取金额特征"""
        features = pd.DataFrame()
        
        if amount_column not in transactions.columns:
            warnings.warn(f"金额列 {amount_column} 不存在")
            return features
        
        # 按节点分组的金额统计
        node_groups = transactions.groupby('txId')
        amount_stats = node_groups[amount_column].agg([
            'count', 'sum', 'mean', 'std', 'min', 'max', 'median'
        ])
        
        # 计算额外的金额特征
        amount_stats['amount_range'] = amount_stats['max'] - amount_stats['min']
        amount_stats['amount_cv'] = amount_stats['std'] / amount_stats['mean']
        amount_stats['amount_cv'] = amount_stats['amount_cv'].fillna(0)
        
        self.transaction_features['amount'] = amount_stats
        return amount_stats


class FeatureEngineer:
    """
    特征工程主类，整合所有特征提取器
    """
    
    def __init__(self):
        self.graph_extractor = GraphFeatureExtractor()
        self.transaction_extractor = TransactionFeatureExtractor()
        self.engineered_features = {}
    
    def engineer_features(self, 
                         data,
                         transactions: Optional[pd.DataFrame] = None,
                         feature_types: List[str] = ['graph', 'transaction']) -> pd.DataFrame:
        """
        工程化特征
        """
        all_features = pd.DataFrame(index=range(data.num_nodes))
        
        if 'graph' in feature_types:
            # 图特征
            centrality_features = self.graph_extractor.extract_node_centrality_features(data)
            neighborhood_features = self.graph_extractor.extract_neighborhood_features(data)
            
            all_features = pd.concat([all_features, centrality_features, neighborhood_features], axis=1)
        
        if 'transaction' in feature_types and transactions is not None:
            # 交易特征
            amount_features = self.transaction_extractor.extract_amount_features(transactions)
            
            # 对齐索引
            if not amount_features.empty:
                all_features = all_features.join(amount_features, how='left')
        
        # 处理缺失值
        all_features = all_features.fillna(0)
        
        self.engineered_features = all_features
        return all_features


# 工厂函数
def create_feature_engineer() -> FeatureEngineer:
    """创建特征工程器"""
    return FeatureEngineer()


def extract_graph_features(data) -> pd.DataFrame:
    """快速提取图特征"""
    engineer = FeatureEngineer()
    return engineer.engineer_features(data, feature_types=['graph'])


def extract_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """快速提取交易特征"""
    engineer = FeatureEngineer()
    return engineer.transaction_extractor.extract_amount_features(transactions)


__all__ = [
    'GraphFeatureExtractor',
    'TransactionFeatureExtractor',
    'FeatureEngineer',
    'create_feature_engineer',
    'extract_graph_features',
    'extract_transaction_features'
]