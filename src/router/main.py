from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.router.config import settings
from src.router.api import completions, admin
from src.router.middleware import setup_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载配置
    settings.load_rules("config/rules.yaml")
    yield
    # 关闭时清理资源


app = FastAPI(
    title="LLM Router Gateway",
    description="智能路由层，实现请求分发到不同层级的LLM模型",
    version="1.0.0",
    lifespan=lifespan,
)

# 设置中间件
setup_middleware(app)

# 注册路由
app.include_router(completions.router, prefix="/v1")
app.include_router(admin.router, prefix="/admin")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "llm-router", "version": "1.0.0"}
