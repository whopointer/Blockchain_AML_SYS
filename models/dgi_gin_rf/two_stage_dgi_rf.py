"""
两阶段DGI+随机森林模型
第一阶段：DGI+GIN自监督学习节点嵌入
第二阶段：随机森林监督分类（使用嵌入+原始特征）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

from .dgi import DGIWithGIN, create_dgi_with_gin
from .random_forest_classifier import DownstreamRandomForest
from .trainer import ImprovedTrainer
from .focal_loss import create_loss_function


class TwoStageDGIRandomForest:
    """
    两阶段 DGI + RandomForest（单大图节点分类专用）

    阶段1：DGI + GIN 自监督学习（用全图，不需要 DataLoader）
    阶段2：抽取全图 embedding，与原始特征拼接，用 mask 切分训练 RF

    标签设定：
      - y 中包含字符串 "unknown"（不参与监督）
      - "1" 或 1 表示 illicit（异常，正类）
      - "2" 或 2 表示 licit（正常，负类）
    最终监督阶段统一二值化：illicit -> 1, licit -> 0, unknown -> -1
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden_channels: int = 128,
        gnn_layers: int = 3,
        rf_n_estimators: int = 200,
        rf_max_depth: int = 15,
        rf_class_weight: Optional[Union[str, Dict[int, float]]] = None,
        rf_auto_class_weight: bool = True,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "two_stage_dgi_rf",
        balance_strategy: str = "none",      # 建议：单大图 + RF 优先用 class_weight，而不是 undersample
        loss_type: str = "dgi_bce",
        unknown_label: str = "unknown",
        illicit_label: Union[int, str] = 1,
        licit_label: Union[int, str] = 2,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.balance_strategy = balance_strategy
        self.loss_type = loss_type

        self.unknown_label = str(unknown_label).strip()
        self.illicit_label = str(illicit_label).strip()
        self.licit_label = str(licit_label).strip()

        self.dgi_model = DGIWithGIN(
            num_features=num_features,
            hidden_channels=hidden_channels,
            num_layers=gnn_layers,
            pooling_strategy="mean",
            corruption_method="feature_shuffle",
        ).to(self.device)

        self.rf_classifier = DownstreamRandomForest(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight=rf_class_weight,
            auto_class_weight=rf_auto_class_weight,
        )

        self.stage1_trained = False
        self.stage2_trained = False
        self.training_history: Dict[str, Any] = {}

        self.logger = logging.getLogger(f"{__name__}.{experiment_name}")

    # -------- label helpers --------
    def _labels_to_str(self, y_any) -> np.ndarray:
        if isinstance(y_any, torch.Tensor):
            y_np = y_any.detach().cpu().numpy()
        else:
            y_np = np.asarray(y_any)
        return np.array([str(v).strip() for v in y_np.reshape(-1)], dtype=object)

    def _binarize_labels(self, y_any) -> np.ndarray:
        y_str = self._labels_to_str(y_any)
        y_bin = np.full((y_str.shape[0],), -1, dtype=int)
        y_bin[y_str == self.licit_label] = 0
        y_bin[y_str == self.illicit_label] = 1
        y_bin[y_str == self.unknown_label] = -1
        return y_bin

    def _mask_to_idx(self, mask: torch.Tensor) -> np.ndarray:
        return torch.nonzero(mask, as_tuple=False).view(-1).detach().cpu().numpy()

    # -------- stage 1: DGI self-supervised --------
    def stage1_self_supervised_training(
        self,
        train_data,
        val_data=None,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 15,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
    ) -> Dict[str, Any]:
        """
        单大图 DGI 训练：train_data 就是一张图（PyG Data）
        val_data 可选，不给就用 train_data 自己做 early-stop 参考
        """
        self.logger.info("开始第一阶段：DGI+GIN 自监督训练（单大图）...")

        train_data = train_data.to(self.device)
        if val_data is None:
            val_data = train_data
        else:
            val_data = val_data.to(self.device)

        optimizer = optim.Adam(self.dgi_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, patience // 2))

        best_val_loss = float("inf")
        patience_counter = 0
        last_train_loss = None

        for epoch in range(num_epochs):
            self.dgi_model.train()
            optimizer.zero_grad()

            pos_scores, neg_scores = self.dgi_model(train_data.x, train_data.edge_index, batch=None)
            loss = self.dgi_model.compute_dgi_loss(pos_scores, neg_scores)

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.dgi_model.parameters(), grad_clip)
            optimizer.step()

            last_train_loss = float(loss.item())

            # validation
            self.dgi_model.eval()
            with torch.no_grad():
                pos_v, neg_v = self.dgi_model(val_data.x, val_data.edge_index, batch=None)
                val_loss = self.dgi_model.compute_dgi_loss(pos_v, neg_v)
                val_loss = float(val_loss.item())

            scheduler.step(val_loss)

            improved = val_loss < best_val_loss - 1e-6
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_stage1_model()
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | train_loss={last_train_loss:.4f} | val_loss={val_loss:.4f}"
                )

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.dgi_model.freeze_encoder()
        self.stage1_trained = True

        stage1_results = {
            "best_val_loss": float(best_val_loss),
            "total_epochs": int(epoch + 1),
            "final_train_loss": float(last_train_loss) if last_train_loss is not None else None,
        }
        self.training_history["stage1"] = stage1_results
        self.logger.info(f"第一阶段完成，best_val_loss={best_val_loss:.4f}")
        return stage1_results

    # -------- extract full-graph embeddings --------
    def extract_embeddings_single_graph(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回：
          embeddings: [N, H]
          features:   [N, F]
          labels_bin: [N]  (unknown=-1, licit=0, illicit=1)
        """
        if not self.stage1_trained:
            raise ValueError("第一阶段训练未完成，请先 stage1_self_supervised_training")

        data = data.to(self.device)
        self.dgi_model.eval()

        with torch.no_grad():
            embeddings = self.dgi_model.gin_encoder.get_node_embeddings(data.x, data.edge_index).detach().cpu().numpy()
            features = data.x.detach().cpu().numpy()
            labels_bin = self._binarize_labels(data.y)

        return embeddings, features, labels_bin

    # -------- stage 2: RF supervised --------
    def stage2_supervised_training_single_graph(
        self,
        data,
        tune_hyperparameters: bool = False,
        find_threshold: bool = True,
        threshold_metric: str = "f1",
        include_unknown_in_threshold: bool = False,
        threshold_recall_min: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        单大图 RF 训练：按 train_mask/val_mask 切分，并过滤 unknown。
        """
        self.logger.info("开始第二阶段：RandomForest 监督训练（单大图）...")

        # Extract embeddings/features and binarize labels on the full graph.
        embeddings, features, labels_bin = self.extract_embeddings_single_graph(data)
        X_all = np.hstack([embeddings, features])

        # Require train/val masks for split on a single graph.
        if not hasattr(data, "train_mask") or data.train_mask is None:
            raise ValueError("data.train_mask 不存在")
        if not hasattr(data, "val_mask") or data.val_mask is None:
            raise ValueError("data.val_mask 不存在")

        # Convert masks to index arrays.
        train_idx = self._mask_to_idx(data.train_mask)
        val_idx_all = self._mask_to_idx(data.val_mask)
        val_idx = val_idx_all

        # filter unknown
        # Drop unknown labels from train/val splits.
        train_idx = train_idx[labels_bin[train_idx] >= 0]
        val_idx = val_idx[labels_bin[val_idx] >= 0]

        # Build tabular train/val data for RF.
        X_train, y_train = X_all[train_idx], labels_bin[train_idx]
        X_val, y_val = X_all[val_idx], labels_bin[val_idx]

        # optional: undersample on tabular (not recommended together with class_weight)
        if self.balance_strategy == "undersample":
            pos = np.where(y_train == 1)[0]
            neg = np.where(y_train == 0)[0]
            if len(pos) > 0 and len(neg) > 0:
                m = min(len(pos), len(neg))
                rng = np.random.default_rng(42)
                pos_s = rng.choice(pos, size=m, replace=False)
                neg_s = rng.choice(neg, size=m, replace=False)
                sel = np.concatenate([pos_s, neg_s])
                rng.shuffle(sel)
                X_train, y_train = X_train[sel], y_train[sel]
                self.logger.info(f"undersample 后训练集分布: {np.bincount(y_train)}")

        # Train RF (with optional hyperparameter tuning).
        rf_results = self.rf_classifier.train(
            X_train, y_train,
            X_val, y_val,
            logger=self.logger,
            tune_hyperparameters=tune_hyperparameters
        )

        # Optionally find optimal threshold on validation data.
        if find_threshold:
            if include_unknown_in_threshold:
                y_val_thr = labels_bin[val_idx_all].copy()
                y_val_thr[y_val_thr < 0] = 0
                X_val_thr = X_all[val_idx_all]
            else:
                y_val_thr = y_val
                X_val_thr = X_val

            thr = self.rf_classifier.find_optimal_threshold(
                X_val_thr,
                y_val_thr,
                metric=threshold_metric,
                logger=self.logger,
                recall_min=threshold_recall_min,
            )
            rf_results["optimal_threshold"] = float(thr)

        # Mark stage2 done and persist artifacts.
        self.stage2_trained = True
        self.training_history["stage2"] = rf_results

        self.save_stage2_model()
        self.logger.info(f"第二阶段完成，val_auc={rf_results.get('val_auc', 0.0):.4f}")
        return rf_results

    # -------- end-to-end --------
    def end_to_end_train_single_graph(
        self,
        data,
        dgi_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 15,
        rf_hyperparameter_tuning: bool = False,
        find_threshold: bool = True,
        threshold_metric: str = "f1",
        include_unknown_in_threshold: bool = False,
        threshold_recall_min: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.logger.info("开始端到端两阶段训练（单大图）...")

        stage1_results = self.stage1_self_supervised_training(
            train_data=data,
            val_data=None,
            num_epochs=dgi_epochs,
            learning_rate=learning_rate,
            patience=patience,
        )

        stage2_results = self.stage2_supervised_training_single_graph(
            data=data,
            tune_hyperparameters=rf_hyperparameter_tuning,
            find_threshold=find_threshold,
            threshold_metric=threshold_metric,
            include_unknown_in_threshold=include_unknown_in_threshold,
            threshold_recall_min=threshold_recall_min,
        )

        self.save_training_results()

        combined = {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "overall_performance": {
                "stage1_val_loss": stage1_results["best_val_loss"],
                "stage2_val_auc": stage2_results.get("val_auc", 0.0),
                "stage2_val_ap": stage2_results.get("val_ap", 0.0),
            },
        }

        self.logger.info("端到端训练完成")
        return combined

    # -------- predict --------
    def predict_single_graph(
        self,
        data,
        mask: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        return_raw_labels_1_2: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回：
          pred: 预测标签（默认还原为 1/2，异常=1，正常=2）
          proba: predict_proba 输出 [N, 2] 或 [N, K]
        """
        if not self.stage2_trained:
            raise ValueError("第二阶段未训练，请先训练 RF 或 load_stage2_model")

        embeddings, features, _ = self.extract_embeddings_single_graph(data)
        X_all = np.hstack([embeddings, features])

        if mask is not None:
            idx = self._mask_to_idx(mask)
            X = X_all[idx]
        else:
            idx = None
            X = X_all

        if hasattr(self.rf_classifier, "predict_with_threshold"):
            pred_bin = self.rf_classifier.predict_with_threshold(X, threshold)
        else:
            pred_bin = self.rf_classifier.predict(X)

        proba = self.rf_classifier.predict_proba(X)

        if return_raw_labels_1_2:
            pred = np.where(pred_bin == 1, 1, 2)
        else:
            pred = pred_bin

        if idx is not None:
            full_pred = np.full((X_all.shape[0],), -1, dtype=int)
            full_pred[idx] = pred
            full_proba = None
            try:
                full_proba = np.zeros((X_all.shape[0], proba.shape[1]), dtype=float)
                full_proba[idx] = proba
            except Exception:
                full_proba = proba
            return full_pred, full_proba

        return pred, proba

    # -------- threshold on val --------
    def find_optimal_threshold_single_graph(self, data, metric: str = "f1") -> float:
        if not self.stage2_trained:
            raise ValueError("第二阶段未训练")

        embeddings, features, labels_bin = self.extract_embeddings_single_graph(data)
        X_all = np.hstack([embeddings, features])

        if not hasattr(data, "val_mask") or data.val_mask is None:
            raise ValueError("data.val_mask 不存在")

        val_idx = self._mask_to_idx(data.val_mask)
        val_idx = val_idx[labels_bin[val_idx] >= 0]

        X_val = X_all[val_idx]
        y_val = labels_bin[val_idx]

        thr = self.rf_classifier.find_optimal_threshold(X_val, y_val, metric=metric, logger=self.logger)
        return float(thr)

    # -------- saving/loading --------
    def save_stage1_model(self):
        stage1_path = self.checkpoint_dir / f"{self.experiment_name}_stage1_dgi.pth"
        torch.save(
            {
                "dgi_state_dict": self.dgi_model.state_dict(),
                "model_config": {
                    "num_features": self.num_features,
                    "hidden_channels": self.hidden_channels,
                    "num_layers": self.dgi_model.gin_encoder.num_layers,
                },
                "label_config": {
                    "unknown_label": self.unknown_label,
                    "illicit_label": self.illicit_label,
                    "licit_label": self.licit_label,
                },
            },
            stage1_path,
        )
        self.logger.info(f"第一阶段模型已保存到: {stage1_path}")

    def save_stage2_model(self):
        stage2_path = self.checkpoint_dir / f"{self.experiment_name}_stage2_rf.joblib"
        self.rf_classifier.save_model(str(stage2_path))
        self.logger.info(f"第二阶段模型已保存到: {stage2_path}")

    def save_training_results(self):
        results_path = self.checkpoint_dir / f"{self.experiment_name}_training_results.json"

        def to_jsonable(v):
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        serializable = {stage: {k: to_jsonable(v) for k, v in res.items()} for stage, res in self.training_history.items()}

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        self.logger.info(f"训练结果已保存到: {results_path}")

    def load_stage1_model(self, model_path: str):
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt.get("dgi_state_dict", {})
        model_state = self.dgi_model.state_dict()

        # 兼容 DataParallel 保存的 module 前缀 + 旧版 norms 结构
        cleaned = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k
            if key in model_state:
                cleaned[key] = v

        incompatible = self.dgi_model.load_state_dict(cleaned, strict=False)
        if incompatible.unexpected_keys:
            self.logger.warning(f"加载 stage1 时忽略的参数: {incompatible.unexpected_keys}")
        if incompatible.missing_keys:
            self.logger.warning(f"加载 stage1 时缺失的参数: {incompatible.missing_keys}")
        self.dgi_model.freeze_encoder()
        self.stage1_trained = True

        label_cfg = ckpt.get("label_config", {})
        if "unknown_label" in label_cfg:
            self.unknown_label = str(label_cfg["unknown_label"]).strip()
        if "illicit_label" in label_cfg:
            self.illicit_label = str(label_cfg["illicit_label"]).strip()
        if "licit_label" in label_cfg:
            self.licit_label = str(label_cfg["licit_label"]).strip()

        self.logger.info(f"第一阶段模型已从 {model_path} 加载")

    def load_stage2_model(self, model_path: str):
        self.rf_classifier.load_model(model_path)
        self.stage2_trained = True
        self.logger.info(f"第二阶段模型已从 {model_path} 加载")

    def load_full_model(self, experiment_name: Optional[str] = None):
        if experiment_name:
            self.experiment_name = experiment_name

        stage1_path = self.checkpoint_dir / f"{self.experiment_name}_stage1_dgi.pth"
        if stage1_path.exists():
            self.load_stage1_model(str(stage1_path))

        stage2_path = self.checkpoint_dir / f"{self.experiment_name}_stage2_rf.joblib"
        if stage2_path.exists():
            self.load_stage2_model(str(stage2_path))

        results_path = self.checkpoint_dir / f"{self.experiment_name}_training_results.json"
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                self.training_history = json.load(f)
            self.logger.info(f"训练结果已从 {results_path} 加载")


def create_two_stage_dgi_rf(**kwargs) -> TwoStageDGIRandomForest:
    """
    工厂函数：创建两阶段DGI+随机森林模型
    """
    return TwoStageDGIRandomForest(**kwargs)
