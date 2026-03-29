import time
import logging
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 记录请求
        logger.info(f"Request {request.method} {request.url.path}")

        response = await call_next(request)

        # 记录响应
        process_time = time.time() - start_time
        logger.info(
            f"Response {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.3f}s"
        )

        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        return response


def setup_middleware(app: FastAPI):
    """配置中间件"""
    app.add_middleware(LoggingMiddleware)
