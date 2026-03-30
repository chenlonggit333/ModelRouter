# LLM智能路由层部署指南

**版本**: 1.0  
**日期**: 2026-03-26  
**适用版本**: Phase 1 MVP

---

## 📋 目录

1. [环境要求](#环境要求)
2. [架构概览](#架构概览)
3. [部署前准备](#部署前准备)
4. [逐步部署](#逐步部署)
5. [配置详解](#配置详解)
6. [验证测试](#验证测试)
7. [监控与日志](#监控与日志)
8. [故障排查](#故障排查)
9. [生产环境建议](#生产环境建议)

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 | 数量 |
|------|---------|---------|------|
| **Router Gateway** | 4核8GB | 8核16GB | 3-5台 |
| **轻量模型(7B)** | 16GB显存/32GB内存 | 24GB显存 | 10-20台 |
| **分类器(7B)** | 16GB显存 | 16GB显存 | 4-8台 |
| **GLM5** | 现有H200*2*8卡 | - | 保持现状 |

### 软件要求

- **操作系统**: Ubuntu 20.04/22.04 LTS 或 CentOS 7/8
- **Python**: 3.10+
- **CUDA**: 11.8+ (如使用GPU)
- **Docker**: 20.10+ (可选)
- **Nginx**: 1.18+ (负载均衡)

### 网络要求

- 内网延迟 < 5ms（Router Gateway到模型服务）
- 带宽: 每台模型服务器 ≥ 1Gbps
- 防火墙开放端口: 8000 (Router), 8000-8010 (模型服务)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        用户/客户端                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Nginx 负载均衡层                             │
│              (SSL终止 + 限流 + 健康检查)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Router Gateway 集群 (3-5台)                       │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ 规则筛选     │ 负载均衡     │ 降级处理                 │ │
│  │ (Level 1)    │ 策略         │ (Circuit Breaker)        │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│轻量模型池  │  │中端模型池 │  │GLM5 (Tier 3) │
│(Tier 1)   │  │(Tier 2)   │  │             │
│10-20台    │  │2-4台      │  │H200*2*8卡   │
│Qwen2.5-7B │  │Qwen2.5-32B│  │             │
└──────────┘  └──────────┘  └──────────────┘
```

---

## 部署前准备

### 1. 准备服务器

确保所有服务器满足硬件要求，并已配置SSH免密登录：

```bash
# 测试服务器连通性
for host in router-01 router-02 router-03 glm5-001 qwen-001 qwen-002; do
    ping -c 1 $host && echo "$host: OK" || echo "$host: FAILED"
done
```

### 2. 创建部署用户

在所有服务器上创建专用部署用户：

```bash
# 在每台服务器上执行
sudo useradd -m -s /bin/bash llm-router
sudo usermod -aG sudo llm-router
sudo passwd llm-router
```

### 3. 安装基础依赖

在所有服务器上安装基础软件：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git nginx redis-server

# CentOS/RHEL
sudo yum install -y python3-pip python3-virtualenv git nginx redis
```

### 4. 配置时间同步

```bash
sudo timedatectl set-timezone Asia/Shanghai
sudo systemctl enable --now systemd-timesyncd
```

---

## 逐步部署

### 第一步：部署Router Gateway（3-5台）

#### 1.1 克隆代码

```bash
# 登录到Router服务器
ssh llm-router@router-01

# 克隆代码
cd ~
git clone https://github.com/chenlonggit333/llm-router.git
cd llm-router
```

#### 1.2 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

#### 1.3 配置环境变量

```bash
cp .env.example .env

# 编辑.env文件
vim .env
```

**Router-01 的示例配置：**

```bash
# GLM5服务地址（主）
GLM5_BASE_URL=http://glm5-001.internal:8000

# 轻量模型地址列表（轮询）
LIGHTWEIGHT_BASE_URLS=http://qwen-001.internal:8000,http://qwen-002.internal:8000,http://qwen-003.internal:8000
LIGHTWEIGHT_MODEL_NAME=qwen2.5-7b

# Router配置
ROUTER_PORT=8000
ROUTER_LOG_LEVEL=INFO

# Redis配置（共享缓存）
REDIS_URL=redis://redis-server.internal:6379/0

# 可选：分类器vLLM地址（如本地部署）
# CLASSIFIER_VLLM_URL=http://localhost:8000/v1
```

#### 1.4 配置路由规则

```bash
# 编辑配置文件
vim config/rules.yaml
```

**生产环境建议配置：**

```yaml
# 简单关键词 - 会被路由到轻量模型
simple_keywords:
  - "你好"
  - "您好"
  - "谢谢"
  - "再见"
  - "什么是"
  - "介绍一下"
  - "解释一下"
  - "简单"
  - "简单说"
  - "hello"
  - "hi"
  - "thanks"
  - "bye"

# 复杂关键词 - 会被路由到GLM5
complex_keywords:
  - "代码"
  - "编写"
  - "开发"
  - "实现"
  - "分析"
  - "推理"
  - "计算"
  - "比较"
  - "优化"
  - "设计"
  - "详细"
  - "深入"
  - "复杂"
  - "完整方案"
  - "架构"
  - "算法"
  - "debug"
  - "调试"
  - "review"
  - "audit"

# 阈值配置
thresholds:
  tier1: 0.3    # 低于此值走轻量模型
  tier2: 0.7    # 高于此值走GLM5

token_count:
  simple_max: 100    # token数<100且无明显复杂关键词=简单
  complex_min: 2000  # token数>2000=复杂
```

#### 1.5 创建日志目录

```bash
mkdir -p logs
echo "logs/" >> .gitignore
```

#### 1.6 启动服务（开发模式）

```bash
# 测试启动
python3 -m uvicorn src.router.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 1.7 配置Systemd服务（生产环境）

```bash
# 创建systemd服务文件
sudo tee /etc/systemd/system/llm-router.service > /dev/null <<EOF
[Unit]
Description=LLM Router Gateway
After=network.target

[Service]
Type=simple
User=llm-router
Group=llm-router
WorkingDirectory=/home/llm-router/llm-router
Environment=PATH=/home/llm-router/llm-router/venv/bin
ExecStart=/home/llm-router/llm-router/venv/bin/uvicorn src.router.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llm-router

[Install]
WantedBy=multi-user.target
EOF

# 重载systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl enable llm-router
sudo systemctl start llm-router

# 查看状态
sudo systemctl status llm-router
```

#### 1.8 验证Router Gateway

```bash
# 测试健康检查
curl http://localhost:8000/health

# 测试API（使用auto路由）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# 强制路由到GLM5
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm5",
    "messages": [{"role": "user", "content": "你好"}]
  }'

# 强制路由到轻量模型
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "light",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

#### 1.9 部署多台Router

在其他Router服务器上重复步骤1.1-1.8，确保配置一致。

---

### 第二步：部署轻量模型池（10-20台）

#### 2.1 安装vLLM

```bash
# 登录到轻量模型服务器
ssh llm-router@qwen-001

# 安装CUDA（如使用GPU）
# 参考: https://developer.nvidia.com/cuda-downloads

# 安装Python依赖
pip install vllm==0.3.0 transformers accelerate

# 或者使用conda
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
```

#### 2.2 下载模型

```bash
# 创建模型目录
mkdir -p /models

# 下载Qwen2.5-7B-Instruct
# 方式1: 使用HuggingFace
python3 << 'PYEOF'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
cache_dir = "/models"

# 下载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
print("Model downloaded successfully!")
PYEOF

# 方式2: 使用modelscope（国内推荐）
pip install modelscope
python3 << 'PYEOF'
from modelscope import snapshot_download

model_dir = snapshot_download(
    'qwen/Qwen2.5-7B-Instruct',
    cache_dir='/models',
    revision='master'
)
print(f"Model downloaded to: {model_dir}")
PYEOF
```

#### 2.3 启动vLLM服务

```bash
# 创建启动脚本
cat > start_vllm.sh << 'EOF'
#!/bin/bash

MODEL_PATH="/models/Qwen2.5-7B-Instruct"
PORT=8000

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-batched-tokens 8192 \
    --port $PORT \
    --host 0.0.0.0 \
    --api-key "${API_KEY:-}" \
    --worker-use-ray \
    --engine-use-ray
EOF

chmod +x start_vllm.sh
```

#### 2.4 配置Systemd服务

```bash
sudo tee /etc/systemd/system/vllm-qwen.service > /dev/null <<EOF
[Unit]
Description=vLLM Qwen2.5-7B Service
After=network.target

[Service]
Type=simple
User=llm-router
Group=llm-router
WorkingDirectory=/home/llm-router
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONPATH=/home/llm-router"
ExecStart=/home/llm-router/start_vllm.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vllm-qwen

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vllm-qwen
sudo systemctl start vllm-qwen
```

#### 2.5 验证模型服务

```bash
# 等待服务启动（首次加载模型需要1-2分钟）
sleep 30

# 测试模型服务
curl http://localhost:8000/v1/models

# 测试chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7
  }'
```

#### 2.6 部署多台轻量模型

在其他轻量模型服务器上重复步骤2.1-2.5，注意修改端口号避免冲突。

---

### 第三步：部署分类器服务（可选，4-8台）

**说明**：如果Router Gateway所在服务器的性能足够，可以直接在同一台服务器上部署分类器。如果需要更高并发，单独部署。

#### 3.1 安装vLLM（与轻量模型相同）

```bash
pip install vllm==0.3.0
```

#### 3.2 启动分类器服务

```bash
# 分类器也使用Qwen2.5-7B-Instruct
MODEL_PATH="/models/Qwen2.5-7B-Instruct"
PORT=8100  # 使用不同端口

python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --port $PORT \
    --host 0.0.0.0
```

#### 3.3 配置Router使用分类器

在Router服务器的`.env`文件中添加：

```bash
CLASSIFIER_VLLM_URL=http://classifier-001.internal:8100/v1
```

---

### 第四步：配置Nginx负载均衡

#### 4.1 安装Nginx

```bash
# 已在第一步安装
# 确认安装
nginx -v
```

#### 4.2 配置Nginx

```bash
sudo tee /etc/nginx/sites-available/llm-router > /dev/null <<EOF
upstream router_backend {
    least_conn;  # 使用最少连接算法
    
    server router-01.internal:8000 weight=5;
    server router-02.internal:8000 weight=5;
    server router-03.internal:8000 weight=5;
    
    keepalive 32;
}

server {
    listen 80;
    server_name router.yourcompany.com;
    
    # 日志配置
    access_log /var/log/nginx/llm-router-access.log;
    error_log /var/log/nginx/llm-router-error.log;
    
    # 限流配置
    limit_req_zone \$binary_remote_addr zone=llm_limit:10m rate=10r/s;
    limit_req zone=llm_limit burst=20 nodelay;
    
    location / {
        proxy_pass http://router_backend;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        
        # 长连接配置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # 缓冲区配置
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    # 健康检查端点
    location /nginx-health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# 启用配置
sudo ln -sf /etc/nginx/sites-available/llm-router /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# 测试配置
sudo nginx -t

# 重载Nginx
sudo systemctl reload nginx
```

#### 4.3 （可选）配置SSL

```bash
# 使用Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx

# 申请证书
sudo certbot --nginx -d router.yourcompany.com

# 自动续期
sudo certbot renew --dry-run
```

---

### 第五步：配置Redis（可选，用于缓存）

#### 5.1 安装Redis

```bash
# 已在第一步安装
# 编辑配置
sudo vim /etc/redis/redis.conf
```

#### 5.2 配置Redis

```bash
# 设置密码
requirepass your_redis_password

# 绑定内网IP
bind 0.0.0.0

# 启用持久化
save 900 1
save 300 10
save 60 10000
```

#### 5.3 启动Redis

```bash
sudo systemctl enable redis
sudo systemctl restart redis

# 测试连接
redis-cli ping
```

---

## 配置详解

### 环境变量说明

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| GLM5_BASE_URL | 是 | - | GLM5服务地址 |
| LIGHTWEIGHT_BASE_URLS | 是 | - | 轻量模型地址列表，逗号分隔 |
| LIGHTWEIGHT_MODEL_NAME | 否 | qwen2.5-7b | 轻量模型名称 |
| ROUTER_PORT | 否 | 8000 | Router监听端口 |
| ROUTER_LOG_LEVEL | 否 | INFO | 日志级别 |
| REDIS_URL | 否 | - | Redis连接地址 |
| CLASSIFIER_VLLM_URL | 否 | - | 分类器vLLM地址 |

### 路由规则说明

```yaml
# config/rules.yaml

# 简单关键词：包含这些词且token数<100，会被路由到轻量模型
simple_keywords:
  - "你好"
  - "谢谢"
  - ...

# 复杂关键词：包含这些词，会被路由到GLM5
complex_keywords:
  - "代码"
  - "分析"
  - ...

# 阈值：控制分类器的行为
thresholds:
  tier1: 0.3  # 复杂度<0.3走轻量模型
  tier2: 0.7  # 复杂度>0.7走GLM5

token_count:
  simple_max: 100    # token数<100视为简单候选
  complex_min: 2000  # token数>2000视为复杂
```

---

## 验证测试

### 1. 健康检查

```bash
# 检查Router健康状态
curl http://router-01:8000/health

# 检查管理接口
curl http://router-01:8000/admin/health

# 查看统计信息
curl http://router-01:8000/admin/stats?time_range=24h
```

### 2. 路由测试

```bash
# 测试1：简单问题应该路由到轻量模型
curl -X POST http://router-01:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Router-Debug: true" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# 预期：route_decision=tier1

# 测试2：代码问题应该路由到GLM5
curl -X POST http://router-01:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Router-Debug: true" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "编写一个快速排序算法"}]
  }'
# 预期：route_decision=tier3

# 测试3：强制指定模型
curl -X POST http://router-01:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm5",
    "messages": [{"role": "user", "content": "你好"}]
  }'
# 预期：使用GLM5
```

### 3. 压力测试

```bash
# 使用ab进行压力测试
ab -n 1000 -c 50 -T 'application/json' \
  -p post_data.json \
  http://router-01:8000/v1/chat/completions

# 或使用wrk
wrk -t12 -c400 -d30s \
  -s stress_test.lua \
  http://router-01:8000/v1/chat/completions
```

### 4. 故障转移测试

```bash
# 停止一台轻量模型服务
ssh qwen-001 "sudo systemctl stop vllm-qwen"

# 发送请求，观察是否自动路由到其他实例
curl -X POST http://router-01:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "light",
    "messages": [{"role": "user", "content": "测试"}]
  }'

# 重新启动服务
ssh qwen-001 "sudo systemctl start vllm-qwen"
```

---

## 监控与日志

### 日志位置

| 组件 | 日志路径 | 说明 |
|------|---------|------|
| Router Gateway | `~/llm-router/logs/router.log` | 应用日志 |
| Router Systemd | `/var/log/syslog` 或 `journalctl -u llm-router` | 系统日志 |
| Nginx | `/var/log/nginx/llm-router-*.log` | 访问日志和错误日志 |
| vLLM | `journalctl -u vllm-qwen` | 模型服务日志 |

### 查看日志

```bash
# 实时查看Router日志
tail -f ~/llm-router/logs/router.log

# 查看Systemd日志
sudo journalctl -u llm-router -f

# 查看Nginx日志
sudo tail -f /var/log/nginx/llm-router-error.log

# 统计请求量
awk '{print $1}' /var/log/nginx/llm-router-access.log | sort | uniq -c | sort -rn
```

### 关键指标监控

```bash
# Router Gateway指标
curl http://router-01:8000/admin/stats

# 模型池状态
curl http://router-01:8000/admin/health

# 系统资源
htop
nvidia-smi  # 如使用GPU
```

---

## 故障排查

### 常见问题

#### 1. Router无法启动

**症状**：`sudo systemctl start llm-router` 失败

**排查**：
```bash
# 查看详细错误
sudo journalctl -u llm-router -n 50

# 检查配置文件
python3 -c "from src.router.config import settings; print('Config OK')"

# 检查端口占用
sudo lsof -i :8000
```

**解决**：
- 检查`.env`文件配置是否正确
- 确保依赖已安装：`pip install -r requirements.txt`
- 检查端口是否被占用

#### 2. 模型服务无法连接

**症状**：请求返回502或连接超时

**排查**：
```bash
# 测试模型服务连通性
curl http://qwen-001:8000/v1/models

# 检查防火墙
sudo iptables -L | grep 8000

# 检查Router配置
grep GLM5_BASE_URL .env
grep LIGHTWEIGHT_BASE_URLS .env
```

**解决**：
- 确保模型服务已启动：`sudo systemctl status vllm-qwen`
- 检查防火墙规则，开放相应端口
- 验证环境变量配置

#### 3. 路由不准确

**症状**：简单问题被路由到GLM5，或复杂问题被路由到轻量模型

**排查**：
```bash
# 查看路由日志
grep "route_decision" logs/router.log

# 测试具体case
curl -X POST http://router-01:8000/v1/chat/completions \
  -H "X-Router-Debug: true" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "你的测试问题"}]}'
```

**解决**：
- 调整`config/rules.yaml`中的关键词和阈值
- 检查分类器vLLM是否正常工作
- 查看`classification_time_ms`是否在合理范围(<100ms)

#### 4. 性能问题

**症状**：响应慢、超时

**排查**：
```bash
# 查看Router性能指标
curl http://router-01:8000/admin/stats

# 查看系统资源
top
iostat -x 1

# 查看Nginx连接数
ss -ant | grep :8000 | wc -l
```

**解决**：
- 增加Router实例数量
- 优化Nginx配置（调整worker_processes和keepalive）
- 检查模型服务是否过载，增加实例
- 启用Redis缓存（如未启用）

#### 5. 内存不足

**症状**：服务被杀（OOM Killer）

**排查**：
```bash
# 查看内存使用
free -h
dmesg | grep -i "killed process"
```

**解决**：
- 增加服务器内存
- 减少vLLM的`--max-model-len`参数
- 启用swap分区（临时方案）

---

## 生产环境建议

### 1. 安全配置

```bash
# 配置防火墙（仅开放必要端口）
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 8000  # 仅允许内网访问模型服务
sudo ufw enable

# 配置API密钥
# 在.env中设置：API_KEY=your_secret_key
# 在Nginx中配置：proxy_set_header Authorization "Bearer your_secret_key";
```

### 2. 备份策略

```bash
# 备份配置文件
tar czvf backup-$(date +%Y%m%d).tar.gz config/ .env

# 定期备份（添加到crontab）
0 2 * * * cd /home/llm-router/llm-router && tar czvf /backup/router-$(date +\%Y\%m\%d).tar.gz config/ .env
```

### 3. 扩容方案

**水平扩容**：
```bash
# 新增Router实例
# 1. 在新服务器上部署Router Gateway
# 2. 添加到Nginx upstream配置
# 3. 重载Nginx

# 新增轻量模型实例
# 1. 在新服务器上部署vLLM
# 2. 更新所有Router的.env中的LIGHTWEIGHT_BASE_URLS
# 3. 重启Router服务
```

### 4. 监控告警

```bash
# 使用Prometheus + Grafana监控
# 1. 安装Prometheus
# 2. 配置抓取Router指标
# 3. 创建Grafana仪表盘

# 或使用简单的脚本监控
#!/bin/bash
while true; do
    if ! curl -sf http://localhost:8000/health > /dev/null; then
        echo "$(date): Router is down!" | mail -s "Alert" admin@company.com
    fi
    sleep 60
done
```

---

## 升级指南

### 升级Router Gateway

```bash
cd ~/llm-router
git pull origin master

# 备份配置
cp .env .env.backup
cp config/rules.yaml config/rules.yaml.backup

# 更新依赖
source venv/bin/activate
pip install -r requirements.txt

# 重启服务
sudo systemctl restart llm-router
```

### 升级模型

```bash
# 停止旧模型服务
sudo systemctl stop vllm-qwen

# 下载新模型
# ...

# 更新启动脚本中的模型路径
vim start_vllm.sh

# 启动新模型
sudo systemctl start vllm-qwen
```

---

## 附录

### A. 常用命令速查

```bash
# 查看所有服务状态
sudo systemctl status llm-router
sudo systemctl status vllm-qwen
sudo systemctl status nginx
sudo systemctl status redis

# 重启所有服务
sudo systemctl restart llm-router
sudo systemctl restart vllm-qwen
sudo systemctl restart nginx

# 查看日志
sudo journalctl -u llm-router -f
tail -f logs/router.log

# 测试API
curl http://localhost:8000/health
curl http://localhost:8000/admin/stats
```

### B. 性能调优参数

**vLLM优化**：
```bash
# 启用GPU内存优化
--gpu-memory-utilization 0.9

# 启用分页注意力
--enable-prefix-caching

# 调整batch大小
--max-num-batched-tokens 8192
--max-num-seqs 256
```

**Router Gateway优化**：
```bash
# 增加worker数量
--workers 8

# 启用keepalive
--timeout-keep-alive 60
```

### C. 联系支持

如有问题，请查看：
- 项目文档：`docs/superpowers/specs/`
- 实施计划：`docs/superpowers/plans/`
- GitHub Issues: https://github.com/chenlonggit333/llm-router/issues

---

**文档版本**: 1.0  
**最后更新**: 2026-03-26  
**作者**: AI Assistant
