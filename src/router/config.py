from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import yaml
from typing import List


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTER_")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://user:pass@localhost/router"


class Thresholds(BaseModel):
    tier1: float = 0.3
    tier2: float = 0.7


class TokenThresholds(BaseModel):
    simple_max: int = 100
    complex_min: int = 2000


class RoutingRules(BaseModel):
    simple_keywords: List[str] = Field(default_factory=list)
    complex_keywords: List[str] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    token_count: TokenThresholds = Field(default_factory=TokenThresholds)


def load_routing_rules(path: Path) -> RoutingRules:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RoutingRules(**data)


class Settings:
    def __init__(self):
        self.server = ServerSettings()
        self.rules = None

    def load_rules(self, rules_path: Path):
        self.rules = load_routing_rules(rules_path)


settings = Settings()
