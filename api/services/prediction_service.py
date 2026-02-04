"""
预测服务 - 推理业务逻辑

职责范围：
- 单笔交易预测
- 批量预测
- 预测结果统计

不负责：
- 依赖注入（由 facade.py 负责）
- 健康检查（由 facade.py 负责）
- 模型管理（由 model_service.py 负责）
- 数据管理（由 data_service.py 负责）
- 缓存管理（由 cache_service.py 负责）
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from .entities import PredictionResult, PredictionSummary
from .rule_engine import RuleEngine


class PredictionService:
    """预测服务：负责推理业务逻辑"""

    def __init__(self,
                 model_service,
                 data_service,
                 cache_service,
                 validation_service):
        self.model_service = model_service
        self.data_service = data_service
        self.cache_service = cache_service
        self.validation_service = validation_service
        self.rule_engine = RuleEngine()
        self.logger = logging.getLogger(__name__)

    @property
    def model(self):
        """获取模型实例"""
        return self.model_service.model

    def _ensure_services_ready(self):
        """检查前置条件"""
        if self.model_service.model is None:
            raise ValueError("模型未加载")
        if not self.data_service.is_loaded:
            raise ValueError("数据未加载")
        if not self.cache_service.is_cache_built:
            raise ValueError("缓存未构建")

    def predict_transactions(self, tx_ids: List[str]) -> List[Dict[str, Any]]:
        """预测指定交易"""
        self._ensure_services_ready()

        thr = self.cache_service.get_threshold()
        pos_proba = self.cache_service.get_cached_proba()
        tx_mapping = self.data_service.get_tx_mapping()

        results = []

        for raw_tx_id in tx_ids:
            tx_key = str(raw_tx_id).strip()

            # 1. 规则引擎路径
            if tx_key not in tx_mapping:
                results.append(self.rule_engine.predict(tx_key))
                continue

            # 2. 模型路径
            idx = int(tx_mapping[tx_key])
            prob = float(pos_proba[idx])
            is_unknown = self.data_service.is_unknown_label(tx_key)

            result_entity = PredictionResult(
                tx_id=tx_key,
                probability=prob,
                threshold=thr,
                is_unknown_label=is_unknown
            )
            results.append(result_entity.to_dict())

        return results

    def batch_predict(self) -> Dict[str, Any]:
        """批量预测"""
        self._ensure_services_ready()

        thr = self.cache_service.get_threshold()
        pos_proba = self.cache_service.get_cached_proba()
        tx_ids = self.data_service.get_all_tx_ids()

        # 兜底：如果 ID 数量和概率数量对不上
        if len(tx_ids) != len(pos_proba):
            self.logger.warning("ID数量与概率数量不匹配，使用生成ID")
            tx_ids = [f"tx_{i}" for i in range(len(pos_proba))]

        # 批量构建实体对象
        entity_list = []
        for i, tx_id in enumerate(tx_ids):
            prob = float(pos_proba[i])
            entity = PredictionResult(
                tx_id=tx_id,
                probability=prob,
                threshold=thr
            )
            entity_list.append(entity)

        # 统计计算
        summary_calc = PredictionSummary(entity_list, thr)
        statistics = summary_calc.calculate()

        return {
            "statistics": statistics,
            "timestamp": datetime.now().isoformat(),
            "cache_built_at": self.cache_service._cache_built_at,
        }

    def get_prediction_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取预测摘要"""
        threshold = self.cache_service.get_threshold()
        entities = [
            PredictionResult(
                tx_id=r.get("tx_id", ""),
                probability=r.get("probability", 0.0),
                threshold=threshold,
                is_unknown_label=bool(r.get("error"))
            )
            for r in results
        ]
        summary_calc = PredictionSummary(entities, threshold)
        return summary_calc.calculate()


__all__ = ['PredictionService']