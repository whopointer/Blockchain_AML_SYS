"""
数据库数据加载服务

从 MySQL 数据库加载地址特征和图数据，用于模型推理
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.db_config import DatabaseConfig
from database.models import SessionLocal

logger = logging.getLogger(__name__)


class DataLoaderService:
    """数据库数据加载服务"""

    def __init__(self):
        self.db_config = DatabaseConfig()

    def get_session(self) -> Session:
        """获取数据库会话"""
        return SessionLocal()

    def load_address_features(self, address: str) -> Optional[Dict[str, Any]]:
        """
        从数据库加载单个地址的特征

        Args:
            address: 区块链地址

        Returns:
            包含特征和元数据的字典，如果没有找到返回 None
        """
        with self.get_session() as session:
            result = session.execute(
                text("""
                    SELECT address, tx_class, step, features
                    FROM blockchain_addresses
                    WHERE address = :address
                """),
                {"address": address}
            ).fetchone()

            if result is None:
                return None

            return {
                "address": result[0],
                "tx_class": result[1],
                "step": result[2],
                "features": json.loads(result[3]) if result[3] else None
            }

    def load_addresses_features(self, addresses: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量加载多个地址的特征

        Args:
            addresses: 地址列表

        Returns:
            地址到特征的映射字典
        """
        if not addresses:
            return {}

        with self.get_session() as session:
            placeholders = ','.join([f":addr{i}" for i in range(len(addresses))])
            params = {f"addr{i}": addr for i, addr in enumerate(addresses)}

            result = session.execute(
                text(f"""
                    SELECT address, tx_class, step, features
                    FROM blockchain_addresses
                    WHERE address IN ({placeholders})
                """),
                params
            )

            features_map = {}
            for row in result:
                features_map[row[0]] = {
                    "address": row[0],
                    "tx_class": row[1],
                    "step": row[2],
                    "features": json.loads(row[3]) if row[3] else None
                }

            return features_map

    def load_address_neighbors(
        self,
        address: str,
        depth: int = 1,
        max_nodes: int = 1000
    ) -> List[str]:
        """
        加载地址的邻居节点（用于构建子图）

        Args:
            address: 目标地址
            depth: 深度（1=直接邻居，2=邻居的邻居）
            max_nodes: 最大节点数

        Returns:
            邻居地址列表
        """
        if depth == 1:
            # 直接邻居
            with self.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT DISTINCT CASE
                            WHEN address_from = :address THEN address_to
                            ELSE address_from
                        END as neighbor
                        FROM transaction_edges
                        WHERE address_from = :address OR address_to = :address
                        LIMIT :limit
                    """),
                    {"address": address, "limit": max_nodes}
                )
                return [row[0] for row in result]

        else:
            # 多度邻居 - 使用递归 CTE
            with self.get_session() as session:
                # 限制深度避免查询过大
                depth = min(depth, 2)

                result = session.execute(
                    text(f"""
                        WITH RECURSIVE neighbors AS (
                            -- 起始节点
                            SELECT address_from as address, 1 as depth
                            FROM transaction_edges
                            WHERE address_to = :address

                            UNION

                            SELECT address_to as address, 1 as depth
                            FROM transaction_edges
                            WHERE address_from = :address

                            UNION ALL

                            SELECT e.address_to as address, n.depth + 1
                            FROM neighbors n
                            JOIN transaction_edges e ON e.address_from = n.address
                            WHERE n.depth < :depth
                        )
                        SELECT DISTINCT address FROM neighbors
                        WHERE address != :address
                        LIMIT :limit
                    """),
                    {"address": address, "depth": depth, "limit": max_nodes}
                )
                return [row[0] for row in result]

    def load_subgraph_data(
        self,
        address: str,
        depth: int = 1,
        max_nodes: int = 1000
    ) -> Dict[str, Any]:
        """
        加载完整的子图数据（包含节点特征和边关系）

        Args:
            address: 目标地址
            depth: 邻居深度
            max_nodes: 最大节点数

        Returns:
            包含节点和边的子图数据
        """
        # 1. 获取所有相关节点
        neighbors = self.load_address_neighbors(address, depth, max_nodes)
        all_addresses = [address] + neighbors

        # 2. 批量加载所有节点的特征
        features_map = self.load_addresses_features(all_addresses)

        # 3. 获取所有边
        with self.get_session() as session:
            placeholders = ','.join([f":addr{i}" for i in range(len(all_addresses))])
            params = {f"addr{i}": addr for i, addr in enumerate(all_addresses)}

            edges_result = session.execute(
                text(f"""
                    SELECT address_from, address_to
                    FROM transaction_edges
                    WHERE address_from IN ({placeholders})
                    AND address_to IN ({placeholders})
                """),
                params
            )

            edges = [(row[0], row[1]) for row in edges_result]

        return {
            "nodes": features_map,
            "edges": edges,
            "root_address": address,
            "total_nodes": len(features_map),
            "total_edges": len(edges)
        }

    def check_address_exists(self, address: str) -> bool:
        """检查地址是否存在于数据库中"""
        with self.get_session() as session:
            result = session.execute(
                text("""
                    SELECT 1 FROM blockchain_addresses
                    WHERE address = :address
                    LIMIT 1
                """),
                {"address": address}
            )
            return result.fetchone() is not None

    def build_pyg_subgraph(self, address: str, depth: int = 1, max_nodes: int = 500):
        """
        构建 PyTorch Geometric 格式的子图数据
        
        Args:
            address: 目标地址
            depth: 邻居深度
            max_nodes: 最大节点数
            
        Returns:
            PyG Data 对象，包含:
                - x: 节点特征 [num_nodes, 165]
                - edge_index: 边索引 [2, num_edges]
                - address_to_idx: 地址到节点索引的映射
        """
        import torch
        from torch_geometric.data import Data
        
        # 1. 获取子图数据
        subgraph_data = self.load_subgraph_data(address, depth, max_nodes)
        
        # 2. 构建地址到索引的映射
        addresses = list(subgraph_data["nodes"].keys())
        address_to_idx = {addr: idx for idx, addr in enumerate(addresses)}
        
        # 3. 构建节点特征矩阵
        num_nodes = len(addresses)
        x = torch.zeros((num_nodes, 165), dtype=torch.float32)
        
        for addr, idx in address_to_idx.items():
            node_data = subgraph_data["nodes"].get(addr)
            if node_data and node_data.get("features"):
                features = node_data["features"]
                if len(features) == 165:
                    x[idx] = torch.tensor(features, dtype=torch.float32)
        
        # 4. 构建边索引
        edges = subgraph_data["edges"]
        edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
        
        for i, (src, dst) in enumerate(edges):
            if src in address_to_idx and dst in address_to_idx:
                edge_index[0, i] = address_to_idx[src]
                edge_index[1, i] = address_to_idx[dst]
        
        # 移除零边
        nonzero_edges = edge_index.sum(0) > 0
        edge_index = edge_index[:, nonzero_edges]
        
        # 5. 创建 PyG Data 对象
        data = Data(x=x, edge_index=edge_index)
        data.address_to_idx = address_to_idx
        data.root_address = address
        
        return data

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self.get_session() as session:
            # 地址数量
            addr_result = session.execute(
                text("SELECT COUNT(*) FROM blockchain_addresses")
            )
            total_addresses = addr_result.scalar()

            # 边数量
            edge_result = session.execute(
                text("SELECT COUNT(*) FROM transaction_edges")
            )
            total_edges = edge_result.scalar()

            # 分类统计
            class_result = session.execute(
                text("""
                    SELECT tx_class, COUNT(*) as cnt
                    FROM blockchain_addresses
                    WHERE tx_class IS NOT NULL
                    GROUP BY tx_class
                """)
            )
            class_distribution = {row[0]: row[1] for row in class_result}

            return {
                "total_addresses": total_addresses,
                "total_edges": total_edges,
                "class_distribution": class_distribution
            }

    def get_all_addresses(self, limit: Optional[int] = None) -> List[str]:
        """
        获取所有地址列表
        
        Args:
            limit: 可选，限制返回数量
            
        Returns:
            地址列表
        """
        with self.get_session() as session:
            if limit:
                result = session.execute(
                    text("SELECT address FROM blockchain_addresses LIMIT :limit"),
                    {"limit": limit}
                )
            else:
                result = session.execute(
                    text("SELECT address FROM blockchain_addresses")
                )
            return [row[0] for row in result]


# 全局单例
_data_loader_service: Optional[DataLoaderService] = None


def get_data_loader_service() -> DataLoaderService:
    """获取数据加载服务实例"""
    global _data_loader_service
    if _data_loader_service is None:
        _data_loader_service = DataLoaderService()
    return _data_loader_service
