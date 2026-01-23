from typing import List, Dict, Any, Tuple

from api.services.prediction_service import PredictionService


class PredictionController:
    """预测控制器类"""

    def __init__(self, checkpoint_dir: str = None, experiment_name: str = 'gnn_dgi_rf_experiment'):
        self.service = PredictionService(checkpoint_dir=checkpoint_dir, experiment_name=experiment_name)

    @property
    def model(self):
        return self.service.model

    def load_model(self) -> bool:
        return self.service.load_model()

    def predict_transactions(self, tx_ids: List[str]) -> List[Dict[str, Any]]:
        return self.service.predict_transactions(tx_ids)

    def batch_predict(self) -> Dict[str, Any]:
        return self.service.batch_predict()

    def get_model_info(self) -> Dict[str, Any]:
        return self.service.get_model_info()

    def validate_input(self, tx_ids: List[str]) -> Tuple[bool, str]:
        return self.service.validate_input(tx_ids)

    def get_prediction_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.service.get_prediction_summary(results)
