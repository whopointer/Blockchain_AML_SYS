"""
随机森林分类器模块
用于DGI预训练后的下游分类任务
"""


from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import torch

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV


class DownstreamRandomForest:
    """
    下游随机森林分类器
    用于基于DGI嵌入的异常检测（适配 Elliptic 单大图节点分类）

    标签约定（你当前数据）：
      - unknown: "unknown"（字符串，无标签，不参与监督）
      - illicit: 1（异常，正类）
      - licit:  2（正常，负类）
    我们会在进入 RF 之前把标签二值化：illicit->1, licit->0，并过滤 unknown。
    """

    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 15,
            min_samples_split: int = 10,
            min_samples_leaf: int = 5,
            random_state: int = 42,
            n_jobs: int = -1,
            class_weight: Optional[Union[str, Dict[int, float]]] = None,
            auto_class_weight: bool = True,
            pos_label: int = 1,  # 二值化后正类恒为 1
            **kwargs,
    ):
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=class_weight,
            bootstrap=True,
            max_features="sqrt",
            **kwargs,
        )
        self.auto_class_weight = auto_class_weight
        self.pos_label = pos_label

        self.is_trained = False
        self.feature_importance = None
        self.training_stats: Dict[str, Any] = {}
        self.optimal_threshold = 0.5

    # -----------------------------
    # Label helpers (string unknown friendly)
    # -----------------------------
    @staticmethod
    def _to_numpy_labels(y_any) -> np.ndarray:
        """
        把各种形式的 y 转成 numpy array（允许 torch.Tensor / list / np.ndarray / object）
        """
        if isinstance(y_any, torch.Tensor):
            # 注意：torch.Tensor 不支持字符串 dtype，但你的 y 可能不是 tensor（可能是 list/np）
            return y_any.detach().cpu().numpy()
        return np.asarray(y_any)

    @staticmethod
    def _normalize_labels_to_str(y_raw: np.ndarray) -> np.ndarray:
        """
        把标签统一成字符串（strip 后），使得 '1'/1 都能一致比较
        """
        # y_raw 可能是数值，也可能是 object/str
        # 统一用 Python str 转换，避免 dtype 陷阱
        y_str = np.array([str(v).strip() for v in y_raw.reshape(-1)], dtype=object)
        return y_str

    @staticmethod
    def _binarize_and_filter(
            y_raw_any,
            idx: np.ndarray,
            unknown_label: str = "unknown",
            illicit_label: Union[int, str] = 1,
            licit_label: Union[int, str] = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对给定 idx 子集：
          - 过滤 unknown_label
          - 把 illicit->1, licit->0
        返回：
          idx_kept: 过滤后的 idx
          y_bin:    与 idx_kept 对齐的 0/1 标签
        """
        y_raw = DownstreamRandomForest._to_numpy_labels(y_raw_any)
        y_str = DownstreamRandomForest._normalize_labels_to_str(y_raw)

        unknown_s = str(unknown_label).strip()
        illicit_s = str(illicit_label).strip()
        licit_s = str(licit_label).strip()

        sub = y_str[idx]
        is_unknown = (sub == unknown_s)
        is_illicit = (sub == illicit_s)
        is_licit = (sub == licit_s)

        keep = (~is_unknown) & (is_illicit | is_licit)
        idx_kept = idx[keep]

        # 对齐 idx_kept 的二值标签
        sub_kept = sub[keep]
        y_bin = np.zeros(sub_kept.shape[0], dtype=int)
        y_bin[sub_kept == illicit_s] = 1
        y_bin[sub_kept == licit_s] = 0

        return idx_kept, y_bin

    # -----------------------------
    # Embedding extraction (single-graph recommended)
    # -----------------------------
    def extract_embeddings_single_graph(
            self,
            dgi_model: torch.nn.Module,
            data,
            device: torch.device,
            logger: Optional[logging.Logger] = None,
            unknown_label: str = "unknown",
            illicit_label: Union[int, str] = 1,
            licit_label: Union[int, str] = 2,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        ✅ 单大图专用：一次性抽取全图 embedding，然后按 train/val/test mask 过滤 unknown 并二值化标签。

        Returns:
          embeddings_all: [N, D]
          labels_all_bin: [N]  (unknown 位置填 -1，其余为 0/1)
          splits: {'train_idx','val_idx','test_idx'} 这些 idx 都已经过滤 unknown 且 y 已二值化可直接用
        """
        dgi_model.eval()
        data = data.to(device)

        with torch.no_grad():
            if hasattr(dgi_model, "get_node_embeddings"):
                emb = dgi_model.get_node_embeddings(data.x, data.edge_index)
            elif hasattr(dgi_model, "gin_encoder"):
                emb = dgi_model.gin_encoder.get_node_embeddings(data.x, data.edge_index)
            else:
                raise AttributeError("dgi_model 没有 get_node_embeddings 或 gin_encoder，无法抽取嵌入")

        embeddings_all = emb.detach().cpu().numpy()

        # 取原始标签（可能是 torch tensor，也可能是 list/np，且可能含字符串 unknown）
        y_raw_any = getattr(data, "y", None)
        if y_raw_any is None:
            raise ValueError("data.y 不存在，无法训练下游 RF")

        y_raw = self._to_numpy_labels(y_raw_any).reshape(-1)
        y_str = self._normalize_labels_to_str(y_raw)

        # 构造 splits idx
        def mask_to_idx(mask: torch.Tensor) -> np.ndarray:
            return torch.nonzero(mask, as_tuple=False).view(-1).detach().cpu().numpy()

        splits: Dict[str, np.ndarray] = {}

        if hasattr(data, "train_mask") and data.train_mask is not None:
            splits["train_idx"] = mask_to_idx(data.train_mask)
        else:
            raise ValueError("单大图节点分类需要 data.train_mask")

        if hasattr(data, "val_mask") and data.val_mask is not None:
            splits["val_idx"] = mask_to_idx(data.val_mask)

        if hasattr(data, "test_mask") and data.test_mask is not None:
            splits["test_idx"] = mask_to_idx(data.test_mask)

        # 生成 labels_all_bin：unknown=-1，其余 0/1
        unknown_s = str(unknown_label).strip()
        illicit_s = str(illicit_label).strip()
        licit_s = str(licit_label).strip()

        labels_all_bin = np.full((y_str.shape[0],), -1, dtype=int)
        labels_all_bin[y_str == licit_s] = 0
        labels_all_bin[y_str == illicit_s] = 1
        # unknown 保持 -1，其它奇怪标签也保持 -1（后续会过滤）

        # 过滤 splits 并给出每个 split 对齐的 y（二值）
        for key, idx in list(splits.items()):
            idx_kept, y_bin = self._binarize_and_filter(
                y_raw_any=y_raw_any,
                idx=idx,
                unknown_label=unknown_label,
                illicit_label=illicit_label,
                licit_label=licit_label,
            )
            splits[key] = idx_kept  # idx 已过滤 unknown
            # 这里不返回 y_bin，因为你可以用 labels_all_bin[splits[key]] 直接取

            if logger:
                dist = {int(k): int(v) for k, v in zip(*np.unique(labels_all_bin[idx_kept], return_counts=True))}
                logger.info(f"[SPLIT] {key}: n={len(idx_kept)}, dist(bin)={dist}")

        if logger:
            raw_dist = {k: int(v) for k, v in zip(*np.unique(y_str, return_counts=True))}
            logger.info(f"[LABEL] raw dist (string): {raw_dist}")
            logger.info(
                f"[EMB] embeddings shape={embeddings_all.shape}, mean={embeddings_all.mean():.4f}, std={embeddings_all.std():.4f}")

        return embeddings_all, labels_all_bin, splits

    # -----------------------------
    # Training / evaluation
    # -----------------------------
    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            logger: Optional[logging.Logger] = None,
            tune_hyperparameters: bool = False,
    ) -> Dict[str, Any]:
        """
        训练随机森林分类器（要求 y_train/y_val 已经是 0/1）
        """
        if logger:
            logger.info("开始训练随机森林分类器...")

        y_train = np.asarray(y_train).astype(int).reshape(-1)
        y_val = np.asarray(y_val).astype(int).reshape(-1)

        # 过滤掉 -1（如果调用者没过滤干净）
        keep_tr = y_train >= 0
        keep_va = y_val >= 0
        if not np.all(keep_tr):
            X_train, y_train = X_train[keep_tr], y_train[keep_tr]
        if not np.all(keep_va):
            X_val, y_val = X_val[keep_va], y_val[keep_va]

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            raise ValueError(f"训练数据只有一个类别: {unique_classes.tolist()}")

        if logger:
            tr_dist = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
            va_dist = {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))}
            logger.info(f"训练集分布(bin): {tr_dist}")
            logger.info(f"验证集分布(bin): {va_dist}")

        if tune_hyperparameters:
            tuning = self.hyperparameter_tuning(X_train, y_train, logger=logger)
            if "best_params" in tuning:
                self.classifier.set_params(**tuning["best_params"])
                if logger:
                    logger.info(f"最佳超参数: {tuning['best_params']}")

        # 类别权重（只在用户没传 class_weight 且 auto_class_weight=True 时计算）
        if self.classifier.class_weight is None and self.auto_class_weight:
            classes = np.unique(y_train)
            cw = compute_class_weight("balanced", classes=classes, y=y_train)
            cw_dict = dict(zip(classes, cw))
            self.classifier.set_params(class_weight=cw_dict)
            if logger:
                logger.info(f"自动计算类别权重: {cw_dict}")
        else:
            if logger:
                logger.info(f"使用已有 class_weight: {self.classifier.class_weight}")

        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        val_pred = self.classifier.predict(X_val)
        val_proba = self.predict_proba_pos(X_val)

        if len(np.unique(y_val)) > 1:
            val_auc = roc_auc_score(y_val, val_proba)
            val_ap = average_precision_score(y_val, val_proba)
        else:
            val_auc = 0.0
            val_ap = 0.0
            if logger:
                logger.warning("验证集只有一个类别，AUC/AP 置为 0.0")

        self.feature_importance = getattr(self.classifier, "feature_importances_", None)
        if self.feature_importance is not None:
            top_features = np.argsort(self.feature_importance)[-10:][::-1]
            top_importance = self.feature_importance[top_features]
        else:
            top_features = np.array([], dtype=int)
            top_importance = np.array([], dtype=float)

        self.training_stats = {
            "val_auc": float(val_auc),
            "val_ap": float(val_ap),
            "n_features": int(X_train.shape[1]),
            "n_samples": int(X_train.shape[0]),
            "top_features": top_features.tolist(),
            "feature_importance": top_importance.tolist(),
        }

        if logger:
            logger.info(f"验证 AUC={val_auc:.4f}, AP={val_ap:.4f}")
            try:
                report = classification_report(
                    y_val, val_pred,
                    target_names=["正常(licit=0)", "异常(illicit=1)"],
                    zero_division=0
                )
                logger.info(f"\n{report}")
            except Exception as e:
                logger.warning(f"分类报告生成失败: {e}")

        return self.training_stats

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        return self.classifier.predict_proba(X)

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        """
        统一取二值化后“正类=1”的概率列（不写死 [:,1]）
        """
        proba = self.predict_proba(X)
        classes = self.classifier.classes_
        if self.pos_label not in classes:
            return np.zeros((proba.shape[0],), dtype=float)
        pos_idx = int(np.where(classes == self.pos_label)[0][0])
        return proba[:, pos_idx]

    # -----------------------------
    # Thresholding
    # -----------------------------
    def find_optimal_threshold(
            self,
            X_val: np.ndarray,
            y_val: np.ndarray,
            metric: str = "f1",
            logger: Optional[logging.Logger] = None,
            thresholds: Optional[np.ndarray] = None,
            recall_min: Optional[float] = None,
    ) -> float:
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        y_val = np.asarray(y_val).astype(int).reshape(-1)
        keep = y_val >= 0
        X_val = X_val[keep]
        y_val = y_val[keep]

        y_proba = self.predict_proba_pos(X_val)

        if thresholds is None:
            thresholds = np.arange(0.05, 0.95, 0.02)

        best_score = -1.0
        best_thr = 0.5
        best_recall = -1.0

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)

            if metric == "f1":
                score = f1_score(y_val, y_pred, average="binary", zero_division=0)
            elif metric == "precision":
                score = precision_score(y_val, y_pred, average="binary", zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, y_pred, average="binary", zero_division=0)
            else:
                raise ValueError(f"不支持的指标: {metric}")

            rec = recall_score(y_val, y_pred, average="binary", zero_division=0)
            if recall_min is not None and rec < recall_min:
                continue

            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_recall = rec

        if best_score < 0 and recall_min is not None:
            # 如果约束过严导致没有可用阈值，则退回到无约束最优
            for thr in thresholds:
                y_pred = (y_proba >= thr).astype(int)
                if metric == "f1":
                    score = f1_score(y_val, y_pred, average="binary", zero_division=0)
                elif metric == "precision":
                    score = precision_score(y_val, y_pred, average="binary", zero_division=0)
                elif metric == "recall":
                    score = recall_score(y_val, y_pred, average="binary", zero_division=0)
                else:
                    raise ValueError(f"不支持的指标: {metric}")
                if score > best_score:
                    best_score = score
                    best_thr = float(thr)
                    best_recall = recall_score(y_val, y_pred, average="binary", zero_division=0)

        self.optimal_threshold = best_thr

        if logger:
            logger.info(
                f"最优阈值={best_thr:.3f}, metric={metric}, score={best_score:.4f}, "
                f"recall={best_recall:.4f}, recall_min={recall_min}; "
                f"proba stats: min={y_proba.min():.4f}, mean={y_proba.mean():.4f}, max={y_proba.max():.4f}"
            )

        return best_thr

    def predict_with_threshold(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        thr = self.optimal_threshold if threshold is None else float(threshold)
        y_proba = self.predict_proba_pos(X)
        return (y_proba >= thr).astype(int)

    # -----------------------------
    # Persistence
    # -----------------------------
    def save_model(self, filepath: str):
        model_data = {
            "classifier": self.classifier,
            "is_trained": self.is_trained,
            "feature_importance": self.feature_importance,
            "training_stats": self.training_stats,
            "optimal_threshold": self.optimal_threshold,
            "pos_label": self.pos_label,
            "auto_class_weight": self.auto_class_weight,
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)
        self.classifier = model_data["classifier"]
        self.is_trained = model_data["is_trained"]
        self.feature_importance = model_data.get("feature_importance")
        self.training_stats = model_data.get("training_stats", {})
        self.optimal_threshold = model_data.get("optimal_threshold", 0.5)
        self.pos_label = model_data.get("pos_label", 1)
        self.auto_class_weight = model_data.get("auto_class_weight", True)

    # -----------------------------
    # Hyperparameter tuning
    # -----------------------------
    def hyperparameter_tuning(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            logger: Optional[logging.Logger] = None,
            scoring: str = "roc_auc",
    ) -> Dict[str, Any]:
        """
        超参数调优（RandomizedSearchCV）
        注意：y_train 必须是 0/1（二值化后），否则 scoring/auc 会出错或语义反了
        """
        y_train = np.asarray(y_train).astype(int).reshape(-1)
        keep = y_train >= 0
        X_train = X_train[keep]
        y_train = y_train[keep]

        param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 15, None],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt"],
        }

        base_rf = RandomForestClassifier(
            random_state=self.classifier.random_state,
            n_jobs=self.classifier.n_jobs,
            class_weight=self.classifier.class_weight,
            bootstrap=True,
        )

        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring=scoring,
            n_jobs=self.classifier.n_jobs,
            verbose=1,
            random_state=self.classifier.random_state,
        )
        search.fit(X_train, y_train)

        if logger:
            logger.info(f"[TUNE] best_score={search.best_score_:.4f} ({scoring})")
            logger.info(f"[TUNE] best_params={search.best_params_}")

        return {
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        }


def _hyperparameter_tuning(self,
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    私有方法：超参数调优
    """
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt']
    }
    
    # 创建基础随机森林
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # 使用RandomizedSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    grid_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # 执行搜索
    grid_search.fit(X_train, y_train)
    
    if logger:
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳分数: {grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    
    def _hyperparameter_tuning(self, 
                              X_train: np.ndarray, 
                              y_train: np.ndarray,
                              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """
        私有方法：超参数调优
        """
        # 定义参数网格
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt']
        }
        
        # 创建基础随机森林
        rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 使用RandomizedSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        grid_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # 执行搜索
        grid_search.fit(X_train, y_train)
        
        if logger:
            logger.info(f"最佳参数: {grid_search.best_params_}")
            logger.info(f"最佳分数: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }


def create_random_forest_classifier(**kwargs) -> DownstreamRandomForest:
    """
    工厂函数：创建随机森林分类器
    """
    return DownstreamRandomForest(**kwargs)
