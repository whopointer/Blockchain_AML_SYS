"""
验证服务 - 单一职责：输入验证

职责范围：
- 验证输入数据
（统计功能已迁移到 entities.py 的 PredictionSummary）
"""

import logging
from typing import List, Dict, Any, Tuple


class ValidationService:
    """验证服务：负责输入验证"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_input(self, tx_ids: List[str]) -> Tuple[bool, str]:
        """验证交易ID列表"""
        if not tx_ids:
            return False, "交易ID列表不能为空"
        if not isinstance(tx_ids, list):
            return False, "交易ID必须是列表格式"
        if len(tx_ids) > 1000:
            return False, "单次预测交易数量不能超过1000"
        for tx_id in tx_ids:
            if not isinstance(tx_id, str):
                return False, "交易ID必须是字符串格式"
            if len(tx_id.strip()) == 0:
                return False, "交易ID不能为空字符串"
        return True, ""

    def validate_tx_id(self, tx_id: str) -> Tuple[bool, str]:
        """验证单个交易ID"""
        if not isinstance(tx_id, str):
            return False, "交易ID必须是字符串格式"
        if len(tx_id.strip()) == 0:
            return False, "交易ID不能为空字符串"
        return True, ""
