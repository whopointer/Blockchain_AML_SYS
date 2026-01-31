"""
缓存服务 - 单一职责：推理缓存和阈值管理

职责范围：
- 构建和缓存推理结果
- 管理推理阈值
- 获取缓存状态
"""

import logging
import os
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import torch


class CacheService:
    """缓存服务：负责推理缓存和阈值管理"""

    def __init__(self, experiment_name: str = "gnn_dgi_rf_experiment"):
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        self._cached_pos_proba: Optional[np.ndarray] = None
        self._cached_threshold: Optional[float] = None
        self._cache_built_at: Optional[str] = None
        self._cache_built = False

    @property
    def is_cache_built(self) -> bool:
        """检查缓存是否已构建"""
        return self._cache_built and self._cached_pos_proba is not None

    def get_threshold(self) -> float:
        """获取推理阈值：优先使用评估校准阈值，其次训练最优阈值，再退回 0.5"""
        if self._cached_threshold is not None:
            return self._cached_threshold

        checkpoint_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")

        eval_threshold = None
        try:
            eval_path = os.path.join(checkpoint_dir, f"{self.experiment_name}_eval_results.json")
            if os.path.exists(eval_path):
                import json
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_results = json.load(f)
                eval_threshold = eval_results.get("calibrated_threshold")
        except Exception:
            eval_threshold = None

        if eval_threshold is not None:
            try:
                thr = float(eval_threshold)
                if 0.0 < thr < 1.0:
                    self.logger.info(f"推理阈值来源=eval_calibrated, value={thr:.3f}")
                    return thr
            except Exception:
                pass

        # 默认阈值
        self.logger.info("推理阈值来源=default, value=0.500")
        return 0.5

    def _get_pos_label_proba(self, probabilities: np.ndarray, pos_label: int = 1) -> np.ndarray:
        """稳健取出"异常(pos_label=1)"那列概率"""
        if probabilities.ndim == 1:
            return probabilities

        try:
            # 假设 probabilities 是二维的，第一列是类别0，第二列是类别1
            if probabilities.shape[1] >= 2:
                return probabilities[:, 1]
            return probabilities[:, 0]
        except Exception:
            return probabilities

    def build_cache(self, model, full_data) -> bool:
        """构建推理缓存"""
        if self.is_cache_built:
            self.logger.info("缓存已存在，跳过构建")
            return True

        if model is None:
            raise ValueError("模型未加载，无法构建缓存")

        self.logger.info("构建整图推理缓存...")

        try:
            preds, probs = model.predict_single_graph(full_data)

            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()

            probs = np.asarray(probs)
            pos_proba = self._get_pos_label_proba(probs, pos_label=1).astype(float)

            self._cached_pos_proba = pos_proba
            self._cached_threshold = self.get_threshold()
            self._cache_built_at = datetime.now().isoformat()
            self._cache_built = True

            self.logger.info(
                f"缓存构建完成: N={len(self._cached_pos_proba)}, thr={self._cached_threshold:.3f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"缓存构建失败: {e}")
            return False

    def get_cached_proba(self) -> Optional[np.ndarray]:
        """获取缓存的概率"""
        return self._cached_pos_proba

    def get_cache_info(self) -> dict:
        """获取缓存信息"""
        return {
            "built": self.is_cache_built,
            "threshold": float(self._cached_threshold) if self._cached_threshold else None,
            "built_at": self._cache_built_at,
            "num_predictions": len(self._cached_pos_proba) if self._cached_pos_proba is not None else None,
        }

    def clear_cache(self):
        """清除缓存"""
        self._cached_pos_proba = None
        self._cached_threshold = None
        self._cache_built_at = None
        self._cache_built = False
        self.logger.info("缓存已清除")
