import pytest
from pathlib import Path
from src.router.config import Settings, RoutingRules, load_routing_rules


class TestSettings:
    def test_settings_load_from_env(self, monkeypatch):
        monkeypatch.setenv("ROUTER_PORT", "8080")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        settings = Settings()
        assert settings.port == 8080
        assert settings.redis_url == "redis://localhost:6379"

    def test_settings_default_values(self):
        settings = Settings()
        assert settings.port == 8000
        assert settings.log_level == "INFO"


class TestRoutingRules:
    def test_load_routing_rules(self, tmp_path):
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text("""
simple_keywords:
  - "你好"
  - "谢谢"
  - "什么是"
complex_keywords:
  - "代码"
  - "分析"
  - "推理"
thresholds:
  tier1: 0.3
  tier2: 0.7
  token_count:
    simple_max: 100
    complex_min: 2000
""")

        rules = load_routing_rules(rules_file)
        assert "你好" in rules.simple_keywords
        assert "代码" in rules.complex_keywords
        assert rules.thresholds.tier1 == 0.3
