"""
领域实体 (Domain Entities)

充血模型：不仅存储数据，还包含基于数据的业务计算逻辑。
将散落的业务规则内聚到实体中。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class PredictionResult:
    """
    预测结果实体
    充血模型：数据和业务逻辑绑定在一起
    """
    tx_id: str
    probability: float
    threshold: float
    is_unknown_label: bool = False
    prediction_source: str = "model"  # 'model' or 'rule_engine'
    rule_reasons: List[str] = field(default_factory=list)

    # --- 业务逻辑下沉到这里 ---

    @property
    def prediction(self) -> int:
        """预测类别：0-正常, 1-可疑"""
        return 1 if self.probability >= self.threshold else 0

    @property
    def is_suspicious(self) -> bool:
        """是否可疑"""
        return self.prediction == 1

    @property
    def confidence(self) -> float:
        """计算置信度"""
        return float(max(self.probability, 1.0 - self.probability))

    @property
    def risk_level(self) -> str:
        """计算风险等级"""
        if self.probability > 0.8:
            return "high"
        elif self.probability >= 0.5:
            return "medium"
        return "low"

    @property
    def error_note(self) -> str:
        """错误说明"""
        if self.is_unknown_label:
            return "该交易在原始数据中为unknown标签（无监督真值），预测仅供参考"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """统一的序列化输出格式"""
        base = {
            "tx_id": self.tx_id,
            "prediction": self.prediction,
            "probability": round(self.probability, 6),
            "is_suspicious": self.is_suspicious,
            "confidence": round(self.confidence, 6),
            "risk_level": self.risk_level,
            "threshold": round(self.threshold, 6),
            "timestamp": datetime.now().isoformat(),
            "error": self.error_note,
        }
        if self.prediction_source == "rule_engine":
            base["rule_based"] = True
            base["reasons"] = self.rule_reasons
        return base


@dataclass
class PredictionSummary:
    """
    统计摘要实体
    将统计计算逻辑封装在实体内部
    """
    results: List[PredictionResult]
    threshold: float

    def calculate(self) -> Dict[str, Any]:
        """计算统计信息"""
        if not self.results:
            return self._empty_stats()

        total = len(self.results)
        suspicious = sum(1 for r in self.results if r.is_suspicious)

        # 风险分布统计
        high = sum(1 for r in self.results if r.risk_level == "high")
        medium = sum(1 for r in self.results if r.risk_level == "medium")
        low = total - high - medium

        return {
            "total_transactions": total,
            "suspicious_count": suspicious,
            "legitimate_count": total - suspicious,
            "suspicious_rate": float(suspicious / total) if total > 0 else 0.0,
            "threshold": self.threshold,
            "risk_distribution": {
                "high_risk": high,
                "medium_risk": medium,
                "low_risk": low,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _empty_stats(self) -> Dict[str, Any]:
        """空统计结果"""
        return {
            "total_transactions": 0,
            "suspicious_count": 0,
            "legitimate_count": 0,
            "suspicious_rate": 0.0,
            "legitimate_rate": 0.0,
            "threshold": self.threshold,
            "risk_distribution": {"high_risk": 0, "medium_risk": 0, "low_risk": 0},
        }
