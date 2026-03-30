#!/bin/bash

# LLM Router 启动脚本

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt -q

# 创建日志目录
mkdir -p logs

# 加载环境变量
if [ -f ".env" ]; then
    echo "Loading environment variables..."
    export $(grep -v '^#' .env | xargs)
fi

# 启动服务
echo "Starting LLM Router Gateway..."
python3 -m uvicorn src.router.main:app --host 0.0.0.0 --port 8000 --workers 4
