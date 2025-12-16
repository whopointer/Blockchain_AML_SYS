"""
预测相关的数据验证模式
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class TransactionPrediction(BaseModel):
    """单个交易预测结果"""
    tx_id: str = Field(..., description="交易ID")
    prediction: int = Field(..., ge=0, le=1, description="预测结果: 0-正常, 1-异常")
    probability: float = Field(..., ge=0.0, le=1.0, description="异常概率")
    is_suspicious: bool = Field(..., description="是否可疑")
    confidence: float = Field(..., ge=0.0, le=1.0, description="预测置信度")
    timestamp: str = Field(..., description="预测时间戳")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """验证时间戳格式"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("时间戳格式无效")


class PredictionRequest(BaseModel):
    """预测请求"""
    tx_ids: List[str] = Field(..., min_items=1, max_items=1000, description="交易ID列表")
    
    @validator('tx_ids')
    def validate_tx_ids(cls, v):
        """验证交易ID列表"""
        if not v:
            raise ValueError("交易ID列表不能为空")
        
        for tx_id in v:
            if not isinstance(tx_id, str):
                raise ValueError("交易ID必须是字符串")
            if len(tx_id.strip()) == 0:
                raise ValueError("交易ID不能为空字符串")
        
        return v


class PredictionResponse(BaseModel):
    """预测响应"""
    results: List[TransactionPrediction] = Field(..., description="预测结果列表")
    total_transactions: int = Field(..., ge=0, description="总交易数")
    suspicious_count: int = Field(..., ge=0, description="可疑交易数")
    timestamp: str = Field(..., description="响应时间戳")
    
    @validator('suspicious_count')
    def validate_suspicious_count(cls, v, values):
        """验证可疑交易数"""
        if 'results' in values and v != sum(1 for r in values['results'] if r.is_suspicious):
            raise ValueError("可疑交易数与结果不匹配")
        return v


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    results: List[TransactionPrediction] = Field(..., description="预测结果列表")
    statistics: Dict[str, Any] = Field(..., description="统计信息")
    timestamp: str = Field(..., description="响应时间戳")


class ModelInfo(BaseModel):
    """模型信息"""
    model_type: str = Field(..., description="模型类型")
    num_features: int = Field(..., gt=0, description="特征数量")
    num_classes: int = Field(..., gt=0, description="类别数量")
    hidden_channels: int = Field(..., gt=0, description="隐藏层维度")
    gnn_layers: int = Field(..., gt=0, description="GNN层数")
    rf_n_estimators: int = Field(..., gt=0, description="随机森林树数量")
    rf_max_depth: int = Field(..., gt=0, description="随机森林最大深度")
    experiment_name: str = Field(..., description="实验名称")
    checkpoint_dir: str = Field(..., description="检查点目录")
    status: str = Field(..., description="模型状态")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误信息")
    timestamp: str = Field(..., description="错误时间戳")


class HealthResponse(BaseModel):
    """健康检查响应"""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="检查时间戳")
    model_loaded: bool = Field(..., description="模型是否已加载")


class StatisticsResponse(BaseModel):
    """统计信息响应"""
    model_config = {"protected_namespaces": ()}
    
    system_status: str = Field(..., description="系统状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    timestamp: str = Field(..., description="统计时间戳")
    version: str = Field(..., description="系统版本")


class RiskDistribution(BaseModel):
    """风险分布"""
    high_risk: int = Field(..., ge=0, description="高风险交易数")
    medium_risk: int = Field(..., ge=0, description="中风险交易数")
    low_risk: int = Field(..., ge=0, description="低风险交易数")


class PredictionSummary(BaseModel):
    """预测摘要"""
    total_transactions: int = Field(..., ge=0, description="总交易数")
    suspicious_count: int = Field(..., ge=0, description="可疑交易数")
    legitimate_count: int = Field(..., ge=0, description="正常交易数")
    suspicious_rate: float = Field(..., ge=0.0, le=1.0, description="可疑交易率")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="平均置信度")
    risk_distribution: RiskDistribution = Field(..., description="风险分布")
    timestamp: str = Field(..., description="摘要时间戳")