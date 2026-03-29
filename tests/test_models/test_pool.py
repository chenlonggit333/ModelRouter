import pytest
from unittest.mock import Mock
from src.models.pool import ModelPool, ModelInstance


class TestModelPool:
    def test_register_instance(self):
        pool = ModelPool()
        instance = ModelInstance(
            id="glm5-001", tier="tier3", host="10.0.0.1", port=8000, max_concurrency=100
        )

        pool.register(instance)

        assert "glm5-001" in pool.instances
        assert pool.get_healthy_instances("tier3") == [instance]

    def test_unregister_instance(self):
        pool = ModelPool()
        instance = ModelInstance(
            id="glm5-001", tier="tier3", host="10.0.0.1", port=8000, max_concurrency=100
        )
        pool.register(instance)

        pool.unregister("glm5-001")

        assert "glm5-001" not in pool.instances

    def test_mark_unhealthy(self):
        pool = ModelPool()
        instance = ModelInstance(
            id="glm5-001", tier="tier3", host="10.0.0.1", port=8000, max_concurrency=100
        )
        pool.register(instance)

        pool.mark_unhealthy("glm5-001")

        healthy = pool.get_healthy_instances("tier3")
        assert instance not in healthy

    def test_get_instances_by_tier(self):
        pool = ModelPool()
        glm5 = ModelInstance(
            id="glm5-001", tier="tier3", host="10.0.0.1", port=8000, max_concurrency=100
        )
        qwen = ModelInstance(
            id="qwen-001", tier="tier1", host="10.0.0.2", port=8000, max_concurrency=100
        )

        pool.register(glm5)
        pool.register(qwen)

        tier3 = pool.get_healthy_instances("tier3")
        tier1 = pool.get_healthy_instances("tier1")

        assert len(tier3) == 1
        assert len(tier1) == 1
        assert tier3[0].id == "glm5-001"
