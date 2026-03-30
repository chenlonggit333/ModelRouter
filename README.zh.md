# LLM智能路由层 (LLM Router)

智能路由层，实现请求分发到不同层级的LLM模型，优化成本和响应速度。

## 功能特性

- **智能路由**: 根据问题复杂度自动选择合适模型（轻量/中端/高端）
- **三级分类**: Level 1规则筛选 + Level 2语义匹配 + Level 3 LLM分类
- **负载均衡**: 支持轮询、最少连接等多种策略
- **降级机制**: 实例故障时自动切换，保证可用性
- **OpenAI兼容API**: 完全兼容OpenAI接口格式

## 架构

```
用户请求 → Router Gateway → [Level 1规则筛选] → [Level 2语义匹配] → [Level 3 LLM分类] → 模型池 → 响应
                                     ↓                    ↓                    ↓
                             60-70%简单请求      10-30%相似查询加速     20-30%复杂请求
                                     ↓                    ↓                    ↓
                             轻量模型(7B)         复用历史决策          GLM5/H100
```

### 三级分类系统

**Level 1 - 规则筛选** (<1ms)
- 关键词匹配，快速判断简单/复杂请求
- 60-70%简单请求直接路由到轻量模型
- 10-15%复杂请求直接路由到GLM5

**Level 2 - 语义匹配** (10-30ms)
- 使用sentence-transformers生成语义向量
- 向量相似度匹配(cosine similarity > 0.85)复用历史决策
- 10-30%请求通过相似查询加速，跳过LLM分类

**Level 3 - LLM分类** (50-100ms)
- 使用轻量模型(如Qwen2.5-7B)进行复杂度分类
- 处理20-30%的边界情况请求
- 返回复杂度评分和置信度

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑.env文件，配置GLM5和轻量模型地址
```

### 3. 启动服务

```bash
./scripts/deploy/start.sh
```

### 4. 测试API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## API文档

启动服务后访问: http://localhost:8000/docs

### 主要接口

- `POST /v1/chat/completions` - OpenAI兼容的聊天完成接口
- `GET /health` - 健康检查
- `GET /admin/stats` - 路由统计信息
- `POST /admin/config` - 更新路由配置

## 项目结构

```
llm-router/
├── src/
│   ├── router/          # Router Gateway
│   ├── classifier/      # 分类器（Level 1 + Level 2 + Level 3）
│   ├── models/          # 模型池和客户端
│   └── common/          # 公共工具
├── tests/               # 测试代码
├── config/              # 配置文件
└── scripts/             # 部署脚本
```

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码风格

```bash
black src/ tests/
isort src/ tests/
```

## 部署

### 📖 详细部署指南

查看完整部署文档：
- **[部署指南](docs/DEPLOYMENT.md)** - 详细的逐步部署说明
- **[部署检查清单](docs/DEPLOYMENT_CHECKLIST.md)** - 确保部署完整的检查清单
- **[设计文档](docs/superpowers/specs/2026-03-26-llm-router-design.md)** - 系统设计说明

### 快速部署

```bash
# 1. 克隆代码
git clone https://github.com/chenlonggit333/llm-router.git
cd llm-router

# 2. 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 配置环境
cp .env.example .env
# 编辑.env配置GLM5和轻量模型地址

# 4. 启动服务
./scripts/deploy/start.sh
```

### 生产环境部署架构

```
                    用户请求
                       │
                       ▼
              ┌─────────────────┐
              │   Nginx负载均衡  │
              │  (3-5台Router)  │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│轻量模型池 │   │中端模型池 │   │ GLM5    │
│(Tier 1)  │   │(Tier 2)  │   │(Tier 3) │
│10-20台   │   │2-4台     │   │H200集群 │
└──────────┘   └──────────┘   └──────────┘
```

**硬件要求：**
- **Router Gateway**: 3-5台，4核8GB+
- **轻量模型(Qwen2.5-7B)**: 10-20台，16GB显存
- **分类器服务**: 4-8台，16GB显存（可与轻量模型共享）
- **GLM5**: 保持现有H200*2*8卡部署

### Docker部署（可选）

```bash
docker build -t llm-router .
docker run -p 8000:8000 --env-file .env llm-router
```

## 许可证

MIT
