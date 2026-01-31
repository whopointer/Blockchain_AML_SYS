"""
预测服务 - 统一入口 Facade

职责范围：
- 组合各子服务
- 提供统一的对外接口
- 流程编排（不再包含业务逻辑）

子服务：
- model_service: 模型生命周期管理
- data_service: 数据集和图结构管理
- cache_service: 推理缓存和阈值管理
- rule_engine: 规则引擎

领域实体：
- PredictionResult: 预测结果（数据和业务逻辑绑定）
- PredictionSummary: 统计摘要（封装统计计算）
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .entities import PredictionResult, PredictionSummary
from .model_service import ModelService
from .data_service import DataService
from .cache_service import CacheService
from .validation_service import ValidationService
from .rule_engine import RuleEngine


class PredictionService:
    """预测服务 Facade：统一入口，负责流程编排"""

    def __init__(self,
                 model_service: ModelService = None,
                 data_service: DataService = None,
                 cache_service: CacheService = None,
                 validation_service: ValidationService = None,
                 checkpoint_dir: Optional[str] = None,
                 experiment_name: str = "gnn_dgi_rf_experiment"):
        # 子服务由外部注入（如 api/__init__.py），若未提供则创建
        self.model_service = model_service if model_service else ModelService(checkpoint_dir, experiment_name)
        self.data_service = data_service if data_service else DataService()
        self.cache_service = cache_service if cache_service else CacheService(experiment_name)
        self.validation_service = validation_service if validation_service else ValidationService()
        self.rule_engine = RuleEngine()

        self.logger = logging.getLogger(__name__)

    @property
    def model(self):
        """获取模型实例"""
        return self.model_service.model

    def _ensure_services_ready(self):
        """内部辅助方法：统一检查前置条件"""
        if self.model_service.model is None:
            raise ValueError("模型未加载")
        if not self.data_service.is_loaded:
            raise ValueError("数据未加载")

        # 自动触发缓存构建
        if not self.cache_service.is_cache_built:
            full_data = self.data_service.get_full_data()
            if full_data is not None:
                self.cache_service.build_cache(self.model_service.model, full_data)

        if not self.cache_service.is_cache_built:
            raise ValueError("缓存构建失败，无法预测")

    def load_model(self) -> bool:
        """加载模型"""
        success = self.model_service.load_model()
        if success:
            # 加载模型后需要加载数据
            self.data_service.load_data()
            # 然后构建缓存
            full_data = self.data_service.get_full_data()
            if full_data is not None:
                self.cache_service.build_cache(self.model_service.model, full_data)
        return success

    def predict_transactions(self, tx_ids: List[str]) -> List[Dict[str, Any]]:
        """预测指定交易"""
        self._ensure_services_ready()

        # 获取上下文数据
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

            # 3. 使用实体类创建结果（逻辑封装在 PredictionResult 内部）
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

        # 使用 DataService 的新接口获取所有交易ID
        tx_ids = self.data_service.get_all_tx_ids()

        # 兜底：如果 ID 数量和概率数量对不上
        if len(tx_ids) != len(pos_proba):
            self.logger.warning("ID数量与概率数量不匹配，使用生成ID")
            tx_ids = [f"tx_{i}" for i in range(len(pos_proba))]

        # 批量构建实体对象用于统计
        entity_list = []
        for i, tx_id in enumerate(tx_ids):
            prob = float(pos_proba[i])
            entity = PredictionResult(
                tx_id=tx_id,
                probability=prob,
                threshold=thr
            )
            entity_list.append(entity)

        # 使用 PredictionSummary 实体进行统计计算
        summary_calc = PredictionSummary(entity_list, thr)
        statistics = summary_calc.calculate()

        return {
            "statistics": statistics,
            "timestamp": datetime.now().isoformat(),
            "cache_built_at": self.cache_service._cache_built_at,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_service.get_model_info()

    def validate_input(self, tx_ids: List[str]) -> tuple:
        """验证输入"""
        return self.validation_service.validate_input(tx_ids)

    def get_prediction_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取预测摘要"""
        threshold = self.cache_service.get_threshold()
        # 将 dict 转换为 PredictionResult 实体列表进行统计
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

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "status": "healthy" if self.model_service.is_loaded else "unhealthy",
            "model_loaded": self.model_service.is_loaded,
            "data_loaded": self.data_service.is_loaded,
            "cache_built": self.cache_service.is_cache_built,
            "timestamp": datetime.now().isoformat(),
        }
