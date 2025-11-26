"""
数据处理工具模块
提供数据清洗、转换、验证等实用工具
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from collections import Counter
import warnings


class DataValidator:
    """
    数据验证器，检查数据质量和完整性
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_graph_data(self, data) -> Dict[str, Any]:
        """验证图数据"""
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 检查基本属性
        if not hasattr(data, 'x') or data.x is None:
            results['errors'].append("缺少节点特征")
            results['is_valid'] = False
        
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            results['errors'].append("缺少边索引")
            results['is_valid'] = False
        
        if results['is_valid']:
            # 检查维度一致性
            num_nodes = data.x.shape[0]
            max_node_idx = data.edge_index.max().item()
            
            if max_node_idx >= num_nodes:
                results['errors'].append(f"边索引超出节点范围: {max_node_idx} >= {num_nodes}")
                results['is_valid'] = False
            
            # 检查边的有效性
            if data.edge_index.shape[0] != 2:
                results['errors'].append("边索引应该是2行")
                results['is_valid'] = False
            
            # 检查特征维度
            if len(data.x.shape) != 2:
                results['warnings'].append("节点特征应该是2维张量")
            
            # 检查图连通性
            if hasattr(data, 'edge_index'):
                unique_nodes = torch.unique(data.edge_index.flatten())
                if len(unique_nodes) < num_nodes:
                    results['warnings'].append(f"图不连通: {len(unique_nodes)}/{num_nodes} 个节点连通")
        
        return results


class FeatureProcessor:
    """
    特征处理器，提供特征工程功能
    """
    
    def __init__(self):
        self.scalers = {}
        self.selectors = {}
        self.pca = {}
    
    def normalize_features(self, 
                          features: pd.DataFrame,
                          method: str = 'standard',
                          feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        特征标准化
        
        Args:
            features: 特征数据
            method: 标准化方法 ('standard', 'minmax', 'robust')
            feature_columns: 要标准化的特征列
        
        Returns:
            标准化后的特征
        """
        if feature_columns is None:
            feature_columns = features.columns.tolist()
        
        normalized_features = features.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"未知的标准化方法: {method}")
        
        normalized_features[feature_columns] = scaler.fit_transform(features[feature_columns])
        self.scalers[method] = scaler
        
        return normalized_features
    
    def select_features(self,
                       features: pd.DataFrame,
                       labels: pd.Series,
                       method: str = 'univariate',
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        特征选择
        
        Args:
            features: 特征数据
            labels: 标签数据
            method: 选择方法 ('univariate', 'mutual_info')
            k: 选择的特征数量
        
        Returns:
            选择后的特征和特征名称
        """
        if method == 'univariate':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"未知的特征选择方法: {method}")
        
        selected_features = selector.fit_transform(features, labels)
        selected_feature_names = features.columns[selector.get_support()].tolist()
        
        self.selectors[method] = selector
        
        return pd.DataFrame(selected_features, columns=selected_feature_names), selected_feature_names


# 工厂函数
def create_data_validator() -> DataValidator:
    """创建数据验证器"""
    return DataValidator()


def create_feature_processor() -> FeatureProcessor:
    """创建特征处理器"""
    return FeatureProcessor()


__all__ = [
    'DataValidator',
    'FeatureProcessor',
    'create_data_validator',
    'create_feature_processor'
]