from abc import ABC, abstractmethod
from typing import List, Optional
from src.models.pool import ModelInstance


class LoadBalancingStrategy(ABC):
    """负载均衡策略基类"""

    @abstractmethod
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        """
        从候选实例中选择一个

        Args:
            instances: 候选实例列表

        Returns:
            选中的实例，如果没有可用实例则返回None
        """
        pass


class RoundRobinStrategy(LoadBalancingStrategy):
    """轮询策略"""

    def __init__(self):
        self._current_index = 0

    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        if not instances:
            return None

        # 选择当前索引的实例
        selected = instances[self._current_index % len(instances)]
        self._current_index += 1
        return selected


class LeastConnectionStrategy(LoadBalancingStrategy):
    """最少连接策略"""

    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        if not instances:
            return None

        # 选择当前负载最小的实例
        return min(instances, key=lambda x: x.current_load)


class QueueDepthStrategy(LoadBalancingStrategy):
    """队列深度策略"""

    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        if not instances:
            return None

        # 选择队列深度最小的实例
        return min(instances, key=lambda x: x.queue_depth)


class LoadBalancer:
    """
    负载均衡器
    支持多种负载均衡策略
    """

    STRATEGIES = {
        "round_robin": RoundRobinStrategy,
        "least_connection": LeastConnectionStrategy,
        "queue_depth": QueueDepthStrategy,
    }

    def __init__(self, strategy: str = "round_robin"):
        strategy_class = self.STRATEGIES.get(strategy, RoundRobinStrategy)
        self._strategy = strategy_class()

    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        """
        从候选实例中选择一个

        Args:
            instances: 候选实例列表

        Returns:
            选中的实例，如果没有可用实例则返回None
        """
        return self._strategy.select(instances)

    def select_with_fallback(
        self,
        primary_instances: List[ModelInstance],
        fallback_instances: List[ModelInstance],
    ) -> Optional[ModelInstance]:
        """
        带降级策略的选择

        优先从primary_instances选择，如果没有可用的则从fallback_instances选择
        """
        # 先从主池选择
        selected = self.select(primary_instances)
        if selected:
            return selected

        # 主池无可用实例，从降级池选择
        return self.select(fallback_instances)
