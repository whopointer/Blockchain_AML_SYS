"""
数据服务 - 单一职责：数据集和图结构管理

职责范围：
- 加载数据集
- 构建图结构
- 管理交易ID映射
"""

import os
import logging
from typing import Dict, Optional
from torch_geometric.data import Data

from data import EllipticDataset


class DataService:
    """数据服务：负责数据集和图结构的加载与管理"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dataset = None
        self.full_data: Optional[Data] = None
        self.tx_mapping: Dict[str, int] = {}
        self.tx_class: Dict[str, str] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """检查数据是否已加载"""
        return self._loaded and self.full_data is not None

    def load_data(self) -> bool:
        """加载数据集和交易ID映射"""
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(project_root, "data")

            self.dataset = EllipticDataset(root=data_path, include_unknown=True)

            # 直接构造全图 Data，避免 dataset[0] 只返回单一时间步子图
            self.full_data = Data(
                x=self.dataset.x,
                edge_index=self.dataset.edge_index,
                y=self.dataset.y,
                time_steps=self.dataset.time_steps,
                num_nodes=self.dataset.x.shape[0],
            )

            if hasattr(self.dataset, "merged_df") and "txId" in self.dataset.merged_df.columns:
                tx_ids = self.dataset.merged_df["txId"].astype(str).tolist()
                self.tx_mapping = {str(tx_id).strip(): idx for idx, tx_id in enumerate(tx_ids)}

                if "class" in self.dataset.merged_df.columns:
                    cls_list = self.dataset.merged_df["class"].astype(str).tolist()
                    self.tx_class = {str(tx_id).strip(): str(c).strip() for tx_id, c in zip(tx_ids, cls_list)}
                else:
                    self.tx_class = {}

                self._loaded = True
                self.logger.info(f"数据加载成功: 节点数={self.full_data.num_nodes}, 映射数={len(self.tx_mapping)}")
                return True
            else:
                self.logger.warning("无法创建交易ID映射：merged_df/txId 不存在")
                return False

        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            self._loaded = False
            return False

    def get_full_data(self) -> Optional[Data]:
        """获取完整图数据"""
        if not self.is_loaded:
            self.load_data()
        return self.full_data

    def get_tx_mapping(self) -> Dict[str, int]:
        """获取交易ID到节点索引的映射"""
        if not self.is_loaded:
            self.load_data()
        return self.tx_mapping

    def get_tx_class(self) -> Dict[str, str]:
        """获取交易ID到类别的映射"""
        if not self.is_loaded:
            self.load_data()
        return self.tx_class

    def get_all_tx_ids(self) -> list:
        """
        获取所有交易ID的列表。
        封装了 pandas 操作，外部不需要知道底层是 DataFrame。
        """
        if not self.is_loaded:
            return []

        # 安全访问内部属性
        if hasattr(self.dataset, "merged_df") and "txId" in self.dataset.merged_df.columns:
            return self.dataset.merged_df["txId"].astype(str).tolist()

        return list(self.tx_mapping.keys())

    def get_tx_class_by_id(self, tx_id: str) -> str:
        """根据交易ID获取类别"""
        return self.tx_class.get(str(tx_id).strip(), "")

    def is_unknown_label(self, tx_id: str) -> bool:
        """检查交易是否为 unknown 标签"""
        return self.get_tx_class_by_id(tx_id).lower() == "unknown"

    def get_num_nodes(self) -> int:
        """获取节点数量"""
        if self.full_data is not None:
            return self.full_data.num_nodes
        return 0
