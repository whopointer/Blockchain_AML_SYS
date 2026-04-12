"""
API路由定义

支持模型：
- GNN: DGI+GIN+RandomForest (PyTorch Geometric)
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel


api_router = APIRouter()
logger = None


class HealthResponseModel(BaseModel):
    status: str
    model_type: str
    model_loaded: bool
    data_loaded: bool
    cache_built: bool
    timestamp: str


class PredictRequest(BaseModel):
    tx_ids: List[str]
    model_type: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    model_type: str
    experiment_name: Optional[str] = None


# ============================================================
# 任务相关模型
# ============================================================
class CreateTaskRequest(BaseModel):
    """创建任务请求（已废弃，请使用 /detect 同步接口）"""

    address: str
    address_type: Optional[str] = "bitcoin"
    neighbor_blocks: Optional[int] = 3
    external_ref: Optional[str] = None
    submitted_by: Optional[str] = None


class DetectRequest(BaseModel):
    """同步检测请求"""

    address: str
    address_type: Optional[str] = "bitcoin"
    model_type: Optional[str] = "gnn"
    neighbor_depth: Optional[int] = 1


class BatchDetectRequest(BaseModel):
    """批量检测请求"""

    addresses: List[str]
    address_type: Optional[str] = "bitcoin"
    model_type: Optional[str] = "gnn"
    neighbor_depth: Optional[int] = 1


class TaskResponse(BaseModel):
    """任务响应"""

    task_id: str
    status: str
    address: str
    address_type: str
    created_at: str
    message: Optional[str] = None


class TaskDetailResponse(BaseModel):
    """任务详情响应"""

    task_id: str
    status: str
    address: str
    address_type: str
    probability: Optional[float] = None
    risk_label: Optional[str] = None
    is_suspicious: Optional[bool] = None
    features: Optional[Dict] = None
    result: Optional[Dict] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class TaskListResponse(BaseModel):
    """任务列表响应"""

    tasks: List[Dict]
    total: int
    limit: int
    offset: int


def get_facade():
    """获取 facade 实例"""
    from . import get_facade as _get

    facade = _get()
    if facade is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    return facade


@api_router.get("/health", response_model=HealthResponseModel, tags=["系统"])
async def health_check():
    """健康检查"""
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        facade = get_facade()
        health = facade.get_health_status()
        return HealthResponseModel(
            status=health["status"],
            model_type=health.get("model_type", "unknown"),
            model_loaded=health["model_loaded"],
            data_loaded=health["data_loaded"],
            cache_built=False,  # 已移除 Redis 缓存
            timestamp=health["timestamp"],
        )
    except Exception as e:
        logger.error(f"健康检查错误: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")


@api_router.post("/predict", tags=["预测"])
async def predict_transactions(request: PredictRequest):
    """
    预测指定交易的异常情况

    使用 DGI+GIN+RandomForest 模型预测

    Args:
        tx_ids: 交易ID列表
        model_type: 模型类型 (gnn)
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        facade = get_facade()
        tx_ids = request.tx_ids

        # 使用模型服务进行预测
        predictions = facade.model_service.predict(tx_ids)

        # 转换结果格式
        results = []
        for pred in predictions:
            if "error" in pred:
                # 地址不存在
                results.append(
                    {
                        "tx_id": pred["address"],
                        "is_suspicious": False,
                        "confidence": 0.0,
                        "risk_level": "unknown",
                        "error": pred.get("error"),
                    }
                )
            else:
                results.append(
                    {
                        "tx_id": pred["address"],
                        "is_suspicious": pred["is_suspicious"],
                        "confidence": pred["probability"],
                        "risk_level": pred["risk_level"],
                    }
                )

        suspicious_count = sum(1 for r in results if r.get("is_suspicious"))
        return {
            "results": results,
            "model_type": "gnn",
            "total_transactions": len(results),
            "suspicious_count": suspicious_count,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@api_router.get("/models", tags=["模型"])
async def list_models():
    """获取支持的模型列表"""
    return {
        "supported_models": ["gnn"],
        "descriptions": {"gnn": "DGI+GIN+RandomForest (PyTorch Geometric)"},
    }


@api_router.post("/model/switch", tags=["模型"])
async def switch_model(request: ModelSwitchRequest):
    """
    切换模型

    Args:
        model_type: 模型类型（gnn）
        experiment_name: 实验名称（可选）
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        facade = get_facade()

        success = facade.switch_model(
            model_type=request.model_type, experiment_name=request.experiment_name
        )

        if success:
            model_info = facade.get_model_info()
            return {
                "success": True,
                "message": f"已切换到 {request.model_type.upper()} 模型",
                "model_info": model_info,
            }
        else:
            raise HTTPException(
                status_code=400, detail=f"模型切换失败: {request.model_type}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"模型切换错误: {e}")
        raise HTTPException(status_code=500, detail="模型切换失败")


@api_router.post("/mode/switch", tags=["模型"])
@api_router.get("/mode", tags=["系统"])
async def get_current_mode():
    """获取当前检测模式"""
    return {
        "mode": "single",
        "model_type": "gnn",
        "description": "DGI+GIN+RandomForest 单模型模式",
    }


@api_router.get("/model/info", tags=["模型"])
async def get_model_info():
    """获取当前模型信息"""
    try:
        facade = get_facade()
        model_info = facade.get_model_info()
        if "error" in model_info:
            raise HTTPException(status_code=503, detail=model_info["error"])
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息错误: {e}")
        raise HTTPException(status_code=500, detail="获取模型信息失败")


@api_router.get("/statistics", tags=["统计"])
async def get_statistics():
    """获取系统统计信息"""
    try:
        facade = get_facade()
        health = facade.get_health_status()
        model_info = facade.get_model_info()
        return {
            "system_status": "running",
            "model_type": health.get("model_type", "unknown"),
            "model_loaded": health["model_loaded"],
            "model_info": model_info,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"获取统计信息错误: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")


@api_router.post("/summary", tags=["统计"])
async def get_prediction_summary(request: Dict):
    """获取预测结果摘要"""
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        facade = get_facade()
        if "results" not in request:
            raise HTTPException(status_code=400, detail="请提供预测结果数据")
        summary = facade.prediction_service.get_prediction_summary(request["results"])
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取预测摘要错误: {e}")
        raise HTTPException(status_code=500, detail="获取预测摘要失败")


# ============================================================
# 同步检测接口（简化版：前端 → 后端 → 数据库 → 模型 → 返回）
# ============================================================


@api_router.post("/detect", tags=["检测"])
async def detect_address(request: DetectRequest):
    """
    同步检测地址风险

    流程：前端 → API → 数据库(特征+图) → 模型预测 → 返回结果

    Args:
        address: 待检测的区块链地址
        address_type: 地址类型 (bitcoin/ethereum)
        model_type: 模型类型 (gnn)
        neighbor_depth: 邻居深度
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        # 1. 从数据库加载数据
        data_loader = get_data_loader_service()

        # 检查地址是否存在
        address_exists = data_loader.check_address_exists(request.address)
        if not address_exists:
            raise HTTPException(
                status_code=404, detail=f"地址不存在于数据库中: {request.address}"
            )

        # 加载地址特征
        address_data = data_loader.load_address_features(request.address)
        if address_data is None or address_data.get("features") is None:
            raise HTTPException(
                status_code=404, detail=f"地址特征数据缺失: {request.address}"
            )

        # 加载子图数据
        subgraph = data_loader.load_subgraph_data(
            request.address, depth=request.neighbor_depth, max_nodes=500
        )

        logger.info(
            f"加载子图: {subgraph['total_nodes']} 节点, {subgraph['total_edges']} 边"
        )

        # 2. 调用模型预测
        facade = get_facade()

        # 准备特征数据（用于模型输入）
        # 这里需要根据模型服务的要求格式化数据
        # 简化处理：直接使用地址特征
        features = address_data["features"]

        # 根据模型类型调用预测
        if request.model_type == "gnn":
            # 使用 GNN 模型预测
            try:
                # 调用模型服务进行预测
                # 这里需要根据实际的模型服务接口调整
                prediction_result = await predict_with_gnn(features, subgraph, facade)
            except Exception as e:
                logger.error(f"GNN模型预测失败: {e}")
                raise HTTPException(status_code=500, detail=f"模型预测失败: {str(e)}")
        else:
            raise HTTPException(
                status_code=400, detail=f"不支持的模型类型: {request.model_type}"
            )

        # 3. 构建返回结果
        probability = prediction_result.get("probability", 0.0)
        is_suspicious = probability > 0.5

        # 确定风险等级
        if probability >= 0.7:
            risk_label = "high"
        elif probability >= 0.4:
            risk_label = "medium"
        elif probability >= 0.2:
            risk_label = "low"
        else:
            risk_label = "normal"

        return {
            "address": request.address,
            "address_type": request.address_type,
            "model_type": request.model_type,
            "probability": probability,
            "is_suspicious": is_suspicious,
            "risk_label": risk_label,
            "original_label": address_data.get("tx_class"),  # 原始标签（如果存在）
            "subgraph_info": {
                "total_nodes": subgraph["total_nodes"],
                "total_edges": subgraph["total_edges"],
                "neighbor_depth": request.neighbor_depth,
            },
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@api_router.post("/batch_detect", tags=["检测"])
async def batch_detect_addresses(request: BatchDetectRequest):
    """
    批量检测多个地址的风险

    流程：对每个地址调用 /detect 逻辑，返回统一格式的结果

    Args:
        addresses: 待检测的区块链地址列表
        address_type: 地址类型 (bitcoin/ethereum)
        model_type: 模型类型 (gnn)
        neighbor_depth: 邻居深度
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        data_loader = get_data_loader_service()

        if not request.addresses or len(request.addresses) == 0:
            raise HTTPException(status_code=400, detail="地址列表不能为空")

        results = []

        # 对每个地址进行检测
        for address in request.addresses:
            try:
                # 检查地址是否存在
                address_exists = data_loader.check_address_exists(address.strip())
                if not address_exists:
                    results.append(
                        {
                            "address": address.strip(),
                            "address_type": request.address_type,
                            "model_type": request.model_type,
                            "probability": 0.0,
                            "is_suspicious": False,
                            "risk_label": "unknown",
                            "error": f"地址不存在于数据库中: {address.strip()}",
                            "subgraph_info": None,
                            "original_label": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                # 加载地址特征
                address_data = data_loader.load_address_features(address.strip())
                if address_data is None or address_data.get("features") is None:
                    results.append(
                        {
                            "address": address.strip(),
                            "address_type": request.address_type,
                            "model_type": request.model_type,
                            "probability": 0.0,
                            "is_suspicious": False,
                            "risk_label": "unknown",
                            "error": f"地址特征数据缺失: {address.strip()}",
                            "subgraph_info": None,
                            "original_label": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                # 加载子图数据
                subgraph = data_loader.load_subgraph_data(
                    address.strip(), depth=request.neighbor_depth, max_nodes=500
                )

                # 调用模型预测
                facade = get_facade()
                features = address_data["features"]

                if request.model_type == "gnn":
                    try:
                        prediction_result = await predict_with_gnn(
                            features, subgraph, facade
                        )
                    except Exception as e:
                        logger.error(f"GNN模型预测失败: {e}")
                        results.append(
                            {
                                "address": address.strip(),
                                "address_type": request.address_type,
                                "model_type": request.model_type,
                                "probability": 0.0,
                                "is_suspicious": False,
                                "risk_label": "unknown",
                                "error": f"模型预测失败: {str(e)}",
                                "subgraph_info": None,
                                "original_label": None,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        continue
                else:
                    results.append(
                        {
                            "address": address.strip(),
                            "address_type": request.address_type,
                            "model_type": request.model_type,
                            "probability": 0.0,
                            "is_suspicious": False,
                            "risk_label": "unknown",
                            "error": f"不支持的模型类型: {request.model_type}",
                            "subgraph_info": None,
                            "original_label": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    continue

                # 确定风险等级
                probability = prediction_result.get("probability", 0.0)
                is_suspicious = probability > 0.5

                if probability >= 0.7:
                    risk_label = "high"
                elif probability >= 0.4:
                    risk_label = "medium"
                elif probability >= 0.2:
                    risk_label = "low"
                else:
                    risk_label = "normal"

                # 添加结果
                results.append(
                    {
                        "address": address.strip(),
                        "address_type": request.address_type,
                        "model_type": request.model_type,
                        "probability": probability,
                        "is_suspicious": is_suspicious,
                        "risk_label": risk_label,
                        "original_label": address_data.get("tx_class"),
                        "subgraph_info": {
                            "total_nodes": subgraph["total_nodes"],
                            "total_edges": subgraph["total_edges"],
                            "neighbor_depth": request.neighbor_depth,
                        },
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                logger.error(f"检测地址 {address} 失败: {e}")
                results.append(
                    {
                        "address": address.strip(),
                        "address_type": request.address_type,
                        "model_type": request.model_type,
                        "probability": 0.0,
                        "is_suspicious": False,
                        "risk_label": "unknown",
                        "error": str(e),
                        "subgraph_info": None,
                        "original_label": None,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # 统计结果
        suspicious_count = sum(
            1 for r in results if r.get("is_suspicious") and not r.get("error")
        )
        success_count = sum(1 for r in results if not r.get("error"))
        error_count = sum(1 for r in results if r.get("error"))

        return {
            "results": results,
            "statistics": {
                "total": len(results),
                "success": success_count,
                "error": error_count,
                "suspicious": suspicious_count,
                "normal": success_count - suspicious_count,
            },
            "model_type": request.model_type,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量检测失败: {str(e)}")


async def predict_with_gnn(features, subgraph, facade):
    """
    使用 GNN 模型进行预测

    这里需要根据实际的模型服务接口进行调整
    简化版本：使用特征直接预测
    """
    import numpy as np

    # 简化处理：将特征转换为 numpy 数组
    # 实际使用时需要根据模型要求格式化
    try:
        feature_array = np.array(features, dtype=np.float32)

        # 调用模型服务的预测接口
        # 这里是一个占位实现，需要根据实际模型服务调整
        # 假设模型服务有 predict_proba 方法

        # 模拟预测结果（实际需要调用真实模型）
        # probability = facade.gnn_model_service.predict_proba(feature_array)

        # 简化：随机生成一个概率（用于测试）
        # 实际部署时需要接入真实模型
        np.random.seed(hash(features[0] if features else "default") % (2**32))
        probability = float(np.random.random() * 0.5)  # 0-0.5 之间的概率

        return {"probability": probability, "model": "gnn"}

    except Exception as e:
        logger.error(f"GNN预测失败: {e}")
        raise


def get_data_loader_service():
    """获取数据加载服务实例"""
    from api.services.data_loader_service import get_data_loader_service as _get

    return _get()


# ============================================================
# 任务管理 API（已简化，请使用 /detect 同步接口）
# ============================================================
@api_router.post("/tasks", response_model=TaskResponse, tags=["任务"])
async def create_task(request: CreateTaskRequest):
    """
    创建检测任务 [已废弃]

    请使用 /api/v1/detect 同步接口进行检测

    此接口已不再使用消息队列，保留仅用于向后兼容
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    # 直接调用同步检测接口
    detect_request = DetectRequest(
        address=request.address,
        address_type=request.address_type,
        model_type="gnn",
        neighbor_depth=request.neighbor_blocks,
    )

    result = await detect_address(detect_request)

    return TaskResponse(
        task_id=f"task_{result['address'][:8]}_{int(datetime.now().timestamp())}",
        status="completed",
        address=result["address"],
        address_type=result["address_type"],
        created_at=datetime.now().isoformat(),
        message="检测完成（同步模式）",
    )


@api_router.get("/tasks/{task_id}", response_model=TaskDetailResponse, tags=["任务"])
async def get_task(task_id: str):
    """
    获取任务详情

    Args:
        task_id: 任务ID
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        from database.models import SessionLocal
        from api.services.task_service import TaskService

        db = SessionLocal()
        try:
            task_service = TaskService(db)
            task_data = task_service.get_task_with_result(task_id)

            if task_data is None:
                raise HTTPException(status_code=404, detail="任务不存在")

            return TaskDetailResponse(**task_data)

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


@api_router.get("/tasks", response_model=TaskListResponse, tags=["任务"])
async def list_tasks(
    status: Optional[str] = Query(None, description="按状态过滤"),
    address: Optional[str] = Query(None, description="按地址搜索"),
    limit: int = Query(50, ge=1, le=100, description="返回数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """
    获取任务列表

    Args:
        status: 任务状态 (pending/processing/completed/failed)
        address: 地址模糊搜索
        limit: 返回数量
        offset: 偏移量
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        from database.models import SessionLocal
        from api.services.task_service import TaskService

        db = SessionLocal()
        try:
            task_service = TaskService(db)
            tasks = task_service.list_tasks(
                status=status, address=address, limit=limit, offset=offset
            )
            total = task_service.count_tasks(status=status)

            return TaskListResponse(
                tasks=[t.to_dict() for t in tasks],
                total=total,
                limit=limit,
                offset=offset,
            )

        finally:
            db.close()

    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@api_router.delete("/tasks/{task_id}", tags=["任务"])
async def delete_task(task_id: str):
    """
    删除任务

    Args:
        task_id: 任务ID
    """
    global logger
    if logger is None:
        logger = __import__("logging").getLogger(__name__)

    try:
        from database.models import SessionLocal
        from api.services.task_service import TaskService

        db = SessionLocal()
        try:
            task_service = TaskService(db)
            success = task_service.delete_task(task_id)

            if not success:
                raise HTTPException(status_code=404, detail="任务不存在")

            return {"message": "任务删除成功", "task_id": task_id}

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")
