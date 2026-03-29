from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import threading


@dataclass
class ModelInstance:
    """模型实例配置"""

    id: str
    tier: str  # tier1, tier2, tier3
    host: str
    port: int
    max_concurrency: int
    current_load: int = 0
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    @property
    def queue_depth(self) -> int:
        """当前队列深度"""
        return self.current_load

    @property
    def is_available(self) -> bool:
        """是否可用"""
        return self.is_healthy and self.current_load < self.max_concurrency

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, ModelInstance):
            return self.id == other.id
        return False


class ModelPool:
    """
    模型池管理
    管理所有模型实例的注册、健康状态、负载情况
    """

    def __init__(self):
        self.instances: Dict[str, ModelInstance] = {}
        self._lock = threading.RLock()

    def register(self, instance: ModelInstance):
        """注册新实例"""
        with self._lock:
            self.instances[instance.id] = instance

    def unregister(self, instance_id: str):
        """注销实例"""
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]

    def get_instance(self, instance_id: str) -> Optional[ModelInstance]:
        """获取指定实例"""
        return self.instances.get(instance_id)

    def get_healthy_instances(self, tier: str) -> List[ModelInstance]:
        """获取指定层级的健康实例"""
        with self._lock:
            return [
                inst
                for inst in self.instances.values()
                if inst.tier == tier and inst.is_healthy
            ]

    def get_available_instances(self, tier: str) -> List[ModelInstance]:
        """获取指定层级可用的实例（健康且未满负载）"""
        with self._lock:
            return [
                inst
                for inst in self.instances.values()
                if inst.tier == tier and inst.is_available
            ]

    def mark_unhealthy(self, instance_id: str):
        """标记实例为不健康"""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].is_healthy = False

    def mark_healthy(self, instance_id: str):
        """标记实例为健康"""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].is_healthy = True
                self.instances[instance_id].last_heartbeat = datetime.now()

    def update_load(self, instance_id: str, delta: int):
        """更新实例负载"""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].current_load += delta
                # 确保负载不会为负
                if self.instances[instance_id].current_load < 0:
                    self.instances[instance_id].current_load = 0

    def get_all_instances(self) -> List[ModelInstance]:
        """获取所有实例"""
        return list(self.instances.values())

    def get_stats(self) -> Dict:
        """获取池统计信息"""
        with self._lock:
            total = len(self.instances)
            healthy = sum(1 for inst in self.instances.values() if inst.is_healthy)
            by_tier = {}
            for inst in self.instances.values():
                if inst.tier not in by_tier:
                    by_tier[inst.tier] = {"total": 0, "healthy": 0}
                by_tier[inst.tier]["total"] += 1
                if inst.is_healthy:
                    by_tier[inst.tier]["healthy"] += 1

            return {
                "total_instances": total,
                "healthy_instances": healthy,
                "unhealthy_instances": total - healthy,
                "by_tier": by_tier,
            }
