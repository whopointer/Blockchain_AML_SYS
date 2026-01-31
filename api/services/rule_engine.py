"""
Rule-based prediction engine interface.
Implement domain rules in RuleEngine.predict using optional context signals.
"""

from datetime import datetime
from typing import Dict, Any, Optional


class RuleEngine:
    """Rule engine for identifiers not in the graph dataset."""

    def predict(self, identifier: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return rule-based prediction for a single identifier.
        context: optional features such as counts, volumes, flags, and graph metrics.
        """
        ctx = context or {}
        reasons = []
        score = 0.0

        # Hard rules (external flags)
        if ctx.get("is_blacklisted") is True:
            reasons.append("blacklisted")
            score = max(score, 0.95)

        if ctx.get("is_mixer") is True or ctx.get("adjacent_to_mixer") is True:
            reasons.append("mixer_association")
            score = max(score, 0.85)

        if ctx.get("adjacent_to_illicit") is True:
            reasons.append("illicit_neighbor")
            score = max(score, 0.8)

        # Behavioral rules (volume / burst / counterparties)
        tx_24h = float(ctx.get("tx_count_24h", 0) or 0)
        counterparties_24h = float(ctx.get("counterparties_24h", 0) or 0)
        total_in_24h = float(ctx.get("total_in_24h", 0) or 0)
        total_out_24h = float(ctx.get("total_out_24h", 0) or 0)
        small_amount_ratio = float(ctx.get("small_amount_ratio", 0) or 0)
        address_age_days = float(ctx.get("address_age_days", 0) or 0)

        if tx_24h >= 200 or counterparties_24h >= 100:
            reasons.append("high_activity")
            score = max(score, 0.7)

        if small_amount_ratio >= 0.85 and tx_24h >= 50:
            reasons.append("smurfing_pattern")
            score = max(score, 0.7)

        if total_in_24h >= 0 and total_out_24h >= 0:
            flow_ratio = (total_out_24h + 1e-9) / (total_in_24h + 1e-9)
            if flow_ratio >= 10 or flow_ratio <= 0.1:
                reasons.append("unbalanced_flows")
                score = max(score, 0.6)

        if address_age_days > 0 and address_age_days <= 7 and tx_24h >= 50:
            reasons.append("new_address_burst")
            score = max(score, 0.65)

        # Graph proximity signals if available
        hops_to_illicit = ctx.get("min_hops_to_illicit")
        if hops_to_illicit is not None and hops_to_illicit <= 2:
            reasons.append("close_to_illicit")
            score = max(score, 0.75)

        if not reasons:
            return {
                "txId": identifier,
                "prediction": 0,
                "probability": 0.5,
                "is_suspicious": False,
                "confidence": 0.5,
                "risk_level": "unknown",
                "timestamp": datetime.now().isoformat(),
                "error": "rule_engine_no_signals",
                "rule_based": True,
                "reasons": [],
            }

        risk_level = "low"
        if score >= 0.7:
            risk_level = "high"
        elif score >= 0.4:
            risk_level = "medium"

        return {
                        "txId": identifier,
                        "prediction": 1 if score >= 0.5 else 0,
                        "probability": round(score, 6),
                        "is_suspicious": bool(score >= 0.5),
                        "confidence": round(max(score, 1.0 - score), 6),
                        "risk_level": risk_level,
                        "timestamp": datetime.now().isoformat(),
                        "error": "",
                        "rule_based": True,
                        "reasons": reasons,
                    }
