import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

from torch_geometric.data import Data

from data import EllipticDataset
from models import create_two_stage_dgi_rf
from api.services.rule_engine import RuleEngine


# 你项目里的依赖，按你项目路径保持不变
# from data.elliptic_dataset import EllipticDataset
# from models.factory import create_two_stage_dgi_rf


class PredictionController:
    """预测控制器类"""

    def __init__(self, checkpoint_dir: str = 'checkpoints', experiment_name: str = 'gnn_dgi_rf_experiment'):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.model = None
        self.logger = logging.getLogger(__name__)

        self.dataset = None
        self.full_data = None

        # txId -> node index
        self.tx_mapping: Dict[str, int] = {}
        # txId -> raw class string ('unknown'/'1'/'2')，用于提示
        self.tx_class: Dict[str, str] = {}

        # 缓存整图推理结果（按 node index 对齐）
        self._cached_pos_proba: Optional[np.ndarray] = None  # shape [N]
        self._cached_threshold: Optional[float] = None
        self._cache_built_at: Optional[str] = None
        self.rule_engine = RuleEngine()

    def load_model(self) -> bool:
        try:
            self.model = create_two_stage_dgi_rf(
                num_features=165,
                num_classes=2,
                hidden_channels=128,
                gnn_layers=3,
                rf_n_estimators=200,
                rf_max_depth=15,
                device='auto',
                checkpoint_dir=self.checkpoint_dir,
                experiment_name=self.experiment_name
            )
            self.model.load_full_model(self.experiment_name)

            self._load_dataset_mapping()
            self._build_full_graph_cache()

            self.logger.info("模型加载成功")
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False

    def _load_dataset_mapping(self):
        """加载数据集和交易ID映射（包含 unknown，保证 txId 能定位到图节点）"""
        try:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(project_root, 'data')

            # 关键：推理服务建议 include_unknown=True（训练阶段再过滤 unknown）
            self.dataset = EllipticDataset(root=data_path, include_unknown=True)

            # 直接构造全图 Data，避免 dataset[0] 只返回单一时间步子图
            self.full_data = Data(
                x=self.dataset.x,
                edge_index=self.dataset.edge_index,
                y=self.dataset.y,
                time_steps=self.dataset.time_steps,
                num_nodes=self.dataset.x.shape[0]
            )

            if hasattr(self.dataset, 'merged_df') and 'txId' in self.dataset.merged_df.columns:
                tx_ids = self.dataset.merged_df['txId'].astype(str).tolist()
                self.tx_mapping = {str(tx_id).strip(): idx for idx, tx_id in enumerate(tx_ids)}

                if 'class' in self.dataset.merged_df.columns:
                    cls_list = self.dataset.merged_df['class'].astype(str).tolist()
                    self.tx_class = {str(tx_id).strip(): str(c).strip() for tx_id, c in zip(tx_ids, cls_list)}
                else:
                    self.tx_class = {}

                self.logger.info(f"已加载 {len(self.tx_mapping)} 个交易ID映射（包含unknown）")
                self.logger.info(
                    f"全图节点数: {self.full_data.num_nodes}, 映射数: {len(self.tx_mapping)}"
                )
            else:
                self.logger.warning("无法创建交易ID映射：merged_df/txId 不存在")
                self.tx_mapping = {}
                self.tx_class = {}

        except Exception as e:
            self.logger.error(f"加载数据集映射失败: {e}")
            self.dataset = None
            self.full_data = None
            self.tx_mapping = {}
            self.tx_class = {}

    def _get_threshold(self) -> float:
        """获取推理阈值：优先使用评估校准阈值，其次训练最优阈值，再退回 0.5"""
        eval_threshold = None
        try:
            import os
            eval_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_eval_results.json")
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

        try:
            thr = float(getattr(self.model.rf_classifier, "optimal_threshold", 0.5))
            if 0.0 < thr < 1.0:
                self.logger.info(f"推理阈值来源=training_optimal, value={thr:.3f}")
                return thr
        except Exception:
            pass
        self.logger.info("推理阈值来源=default, value=0.500")
        return 0.5

    def _get_pos_label_proba(self, probabilities: np.ndarray, pos_label: int = 1) -> np.ndarray:
        """
        稳健取出“异常(pos_label=1)”那列概率，不再盲取 [:,1]
        """
        if probabilities.ndim == 1:
            return probabilities

        try:
            rf = self.model.rf_classifier.classifier
            classes = rf.classes_
            pos_idx = int(np.where(classes == pos_label)[0][0])
        except Exception:
            # 兜底（不推荐但比崩好）
            pos_idx = 1 if probabilities.shape[1] > 1 else 0

        return probabilities[:, pos_idx]

    def _build_full_graph_cache(self, force: bool = False):
        """
        构建整图推理缓存：
        直接把 full_data 放进 list 传给 model.predict，避免 DataLoader 套娃导致 batch=DataLoader
        """
        if (self._cached_pos_proba is not None) and (not force):
            return

        if self.model is None:
            raise ValueError("模型未加载，无法构建缓存")
        if self.full_data is None:
            raise ValueError("整图数据未加载，无法构建缓存")

        self.logger.info("构建整图推理缓存（一次性整图推理）...")

        preds, probs = self.model.predict_single_graph(self.full_data)

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        probs = np.asarray(probs)
        pos_proba = self._get_pos_label_proba(probs, pos_label=1).astype(float)

        self._cached_pos_proba = pos_proba
        self._cached_threshold = self._get_threshold()
        self._cache_built_at = datetime.now().isoformat()

        self.logger.info(
            f"缓存完成: N={len(self._cached_pos_proba)}, thr={self._cached_threshold:.3f}, built_at={self._cache_built_at}"
        )

    def predict_transactions(self, tx_ids: List[str]) -> List[Dict[str, Any]]:
        if self.model is None:
            raise ValueError("模型未加载")
        if self.dataset is None or self.full_data is None or not self.tx_mapping:
            raise ValueError("数据集/映射未加载，无法进行交易预测")

        self._build_full_graph_cache()

        thr = float(self._cached_threshold if self._cached_threshold is not None else 0.5)
        pos_proba = self._cached_pos_proba
        assert pos_proba is not None

        results_map: Dict[str, Dict[str, Any]] = {}

        for raw_tx_id in tx_ids:
            tx_key = str(raw_tx_id).strip()

            if tx_key not in self.tx_mapping:
                results_map[tx_key] = self.rule_engine.predict(tx_key)
                continue

            idx = int(self.tx_mapping[tx_key])
            prob = float(pos_proba[idx])
            pred = int(prob >= thr)
            confidence = float(max(prob, 1.0 - prob))

            if prob > 0.8:
                risk_level = 'high'
            elif prob >= 0.5:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            raw_cls = self.tx_class.get(tx_key, "")
            note = ""
            if raw_cls.lower() == "unknown":
                note = "该交易在原始数据中为unknown标签（无监督真值），预测仅供参考"

            results_map[tx_key] = {
                'tx_id': tx_key,
                'prediction': pred,
                'probability': prob,
                'is_suspicious': bool(pred == 1),
                'confidence': confidence,
                'risk_level': risk_level,
                'threshold': thr,
                'timestamp': datetime.now().isoformat(),
                'error': note
            }

        return [results_map[str(t).strip()] for t in tx_ids]

    def batch_predict(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("模型未加载")
        if self.dataset is None or self.full_data is None:
            raise ValueError("数据集未加载")

        self._build_full_graph_cache()
        thr = float(self._cached_threshold if self._cached_threshold is not None else 0.5)
        pos_proba = self._cached_pos_proba
        assert pos_proba is not None

        if hasattr(self.dataset, 'merged_df') and 'txId' in self.dataset.merged_df.columns:
            tx_ids = self.dataset.merged_df['txId'].astype(str).tolist()
        else:
            tx_ids = [f'tx_{i}' for i in range(len(pos_proba))]

        preds = (pos_proba >= thr).astype(int)

        results = []
        for i, tx_id in enumerate(tx_ids):
            prob = float(pos_proba[i])
            pred = int(preds[i])
            confidence = float(max(prob, 1.0 - prob))

            if prob > 0.8:
                risk_level = 'high'
            elif prob >= 0.5:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            results.append({
                'tx_id': str(tx_id).strip(),
                'prediction': pred,
                'probability': prob,
                'is_suspicious': bool(pred == 1),
                'confidence': confidence,
                'risk_level': risk_level
            })

        total = len(preds)
        suspicious = int(preds.sum())
        legitimate = total - suspicious

        return {
            'results': results,
            'statistics': {
                'total_transactions': total,
                'suspicious_count': suspicious,
                'legitimate_count': legitimate,
                'suspicious_rate': float(suspicious / total) if total > 0 else 0.0,
                'legitimate_rate': float(legitimate / total) if total > 0 else 0.0,
                'threshold': thr,
                'cache_built_at': self._cache_built_at
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {'error': '模型未加载'}
        return {
            'model_type': 'GNN+DGI+RandomForest',
            'num_features': 165,
            'num_classes': 2,
            'hidden_channels': 128,
            'gnn_layers': 3,
            'rf_n_estimators': 200,
            'rf_max_depth': 15,
            'experiment_name': self.experiment_name,
            'checkpoint_dir': self.checkpoint_dir,
            'status': 'loaded',
            'threshold': float(self._cached_threshold) if self._cached_threshold is not None else None,
            'cache_built_at': self._cache_built_at
        }

    def validate_input(self, tx_ids: List[str]) -> Tuple[bool, str]:
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

    def get_prediction_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {'error': '没有预测结果'}

        total = len(results)
        suspicious = sum(1 for r in results if r.get('is_suspicious', False))
        legitimate = total - suspicious
        avg_conf = sum(float(r.get('confidence', 0.0)) for r in results) / total if total > 0 else 0.0

        high_risk = sum(1 for r in results if float(r.get('probability', 0.0)) > 0.8)
        medium_risk = sum(1 for r in results if 0.5 <= float(r.get('probability', 0.0)) <= 0.8)
        low_risk = total - high_risk - medium_risk

        return {
            'total_transactions': total,
            'suspicious_count': suspicious,
            'legitimate_count': legitimate,
            'suspicious_rate': float(suspicious / total) if total > 0 else 0.0,
            'average_confidence': float(avg_conf),
            'risk_distribution': {
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            },
            'threshold': float(self._cached_threshold) if self._cached_threshold is not None else None,
            'timestamp': datetime.now().isoformat()
        }
