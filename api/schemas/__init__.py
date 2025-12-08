"""
API数据验证模式
"""

from .prediction_schemas import PredictionRequest, PredictionResponse, BatchPredictionResponse

__all__ = ['PredictionRequest', 'PredictionResponse', 'BatchPredictionResponse']