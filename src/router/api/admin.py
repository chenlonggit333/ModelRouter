from fastapi import APIRouter, Query
from typing import Optional
from pydantic import BaseModel

router = APIRouter()


class ConfigUpdate(BaseModel):
    tier1_threshold: Optional[float] = None
    tier2_threshold: Optional[float] = None


# 模拟统计数据（实际应从Prometheus或数据库读取）
_mock_stats = {
    "total_requests": 100000,
    "routing_distribution": {"tier1": 65000, "tier2": 20000, "tier3": 15000},
    "avg_latency_ms": {
        "classification": 35,
        "tier1": 800,
        "tier2": 3000,
        "tier3": 8000,
    },
    "accuracy": 0.92,
    "error_rate": 0.001,
}


@router.get("/stats")
async def get_stats(
    time_range: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    granularity: Optional[str] = Query("hour", regex="^(minute|hour|day)$"),
):
    """
    获取路由统计信息

    Args:
        time_range: 时间范围 (1h, 24h, 7d, 30d)
        granularity: 粒度 (minute, hour, day)
    """
    return _mock_stats


@router.get("/health")
async def admin_health():
    """详细健康检查"""
    return {
        "status": "healthy",
        "components": {
            "gateway": "up",
            "classifier": "up",
            "tier1_pool": "up",
            "tier2_pool": "up",
            "tier3_pool": "up",
        },
        "metrics": {"qps": 1500, "queue_depth": 12},
    }


@router.post("/config")
async def update_config(config: ConfigUpdate):
    """更新路由配置"""
    # 实际应更新配置并持久化
    return {"status": "updated", "config": config.dict(exclude_unset=True)}
