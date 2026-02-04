"""
API路由定义
定义所有API端点的路由和处理逻辑
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import logging

from .schemas.prediction_schemas import (
    PredictionRequest, PredictionResponse,
    ModelInfo, ErrorResponse, HealthResponse, StatisticsResponse
)

# 创建蓝图
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

logger = logging.getLogger(__name__)


def get_facade():
    """从 app config 获取 ServiceFacade"""
    return current_app.config.get('SERVICE_FACADE')


def create_response(data, status_code=200):
    """创建统一的响应格式"""
    response = jsonify(data)
    response.status_code = status_code
    return response


def handle_error(error_msg, status_code=400):
    """处理错误响应"""
    error_response = ErrorResponse(
        error=error_msg,
        timestamp=datetime.now().isoformat()
    )
    return create_response(error_response.dict(), status_code)


@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        facade = get_facade()
        health = facade.get_health_status()
        response = HealthResponse(
            status=health['status'],
            timestamp=health['timestamp'],
            model_loaded=health['model_loaded']
        )
        return create_response(response.dict())
    except Exception as e:
        logger.error(f"健康检查错误: {e}")
        return handle_error("健康检查失败", 500)


@api_bp.route('/predict', methods=['POST'])
def predict_transactions():
    """预测指定交易的异常情况"""
    try:
        data = request.get_json()
        if not data:
            return handle_error("请求数据不能为空")

        try:
            prediction_request = PredictionRequest(**data)
        except Exception as e:
            return handle_error(f"输入验证失败: {str(e)}")

        facade = get_facade()
        results = facade.prediction_service.predict_transactions(
            prediction_request.tx_ids
        )

        suspicious_count = sum(1 for r in results if r['is_suspicious'])
        response = PredictionResponse(
            results=results,
            total_transactions=len(results),
            suspicious_count=suspicious_count,
            timestamp=datetime.now().isoformat()
        )

        return create_response(response.dict())

    except Exception as e:
        logger.error(f"预测错误: {e}")
        return handle_error("预测失败", 500)


@api_bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测整个数据集"""
    try:
        facade = get_facade()
        result = facade.prediction_service.batch_predict()

        response = {
            "statistics": result['statistics'],
            "timestamp": result['timestamp'],
        }

        return create_response(response)

    except Exception as e:
        logger.error(f"批量预测错误: {e}")
        return handle_error("批量预测失败", 500)


@api_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    try:
        facade = get_facade()
        model_info = facade.get_model_info()

        if 'error' in model_info:
            return handle_error(model_info['error'], 503)

        response = ModelInfo(**model_info)
        return create_response(response.dict())

    except Exception as e:
        logger.error(f"获取模型信息错误: {e}")
        return handle_error("获取模型信息失败", 500)


@api_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """获取系统统计信息"""
    try:
        facade = get_facade()
        health = facade.get_health_status()
        response = StatisticsResponse(
            system_status='running',
            model_loaded=health['model_loaded'],
            timestamp=datetime.now().isoformat(),
            version='1.0.0'
        )
        return create_response(response.dict())

    except Exception as e:
        logger.error(f"获取统计信息错误: {e}")
        return handle_error("获取统计信息失败", 500)


@api_bp.route('/summary', methods=['POST'])
def get_prediction_summary():
    """获取预测结果摘要"""
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return handle_error("请提供预测结果数据")

        results = data['results']
        facade = get_facade()
        summary = facade.prediction_service.get_prediction_summary(results)

        if 'error' in summary:
            return handle_error(summary['error'])

        return create_response(summary)

    except Exception as e:
        logger.error(f"获取预测摘要错误: {e}")
        return handle_error("获取预测摘要失败", 500)


# 错误处理
@api_bp.errorhandler(404)
def not_found(error):
    return handle_error("端点未找到", 404)


@api_bp.errorhandler(405)
def method_not_allowed(error):
    return handle_error("请求方法不允许", 405)


@api_bp.errorhandler(500)
def internal_error(error):
    return handle_error("内部服务器错误", 500)


def register_routes(app):
    """注册路由到应用"""
    app.register_blueprint(api_bp)
    logger.info("API路由注册完成")