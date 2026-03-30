# LLM智能路由层 (LLM Router)

智能路由层，实现请求分发到不同层级的LLM模型，优化成本和响应速度。

## 功能特性

- **智能路由**: 根据问题复杂度自动选择最优模型层级
- **三级分类架构**:
  - **Level 1**: 规则筛选 (<1ms) - 路由60-70%的简单查询
  - **Level 2**: 语义匹配 (10-30ms) - 通过历史决策复用加速10-30%的查询
  - **Level 3**: LLM分类 (50-100ms) - 处理20-30%的复杂边界情况
- **向量相似度搜索**: 使用sentence-transformers (all-MiniLM-L6-v2) 进行语义匹配
  - 余弦相似度阈值: 0.85（可配置）
  - 线程安全的内存向量存储，使用asyncio.Lock
  - 自动存储路由决策供未来相似查询复用
- **负载均衡**: 支持轮询、最少连接、队列深度等多种策略
- **降级机制**: 实例故障时自动切换；Level 2失败时优雅降级到Level 3
- **OpenAI兼容API**: 完全兼容OpenAI接口格式，可直接替换
- **生产就绪**: 线程安全、完善的错误处理、详细日志记录
- **高性能**: 集群支持3000-5000 QPS

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

#### 聊天完成

```bash
POST /v1/chat/completions
```

**请求示例：**
```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000
}
```

**响应示例：**
```json
{
  "id": "chatcmpl-abc123",
  "model": "qwen2.5-7b",
  "choices": [...],
  "usage": {...},
  "router_info": {
    "complexity_score": 0.25,
    "route_decision": "tier1",
    "classification_time_ms": 45,
    "routing_path": ["level1_rules"]
  }
}
```

**路由路径说明：**
- `["level1_rules"]` - Level 1规则直接匹配
- `["level1_rules", "level2_embedding"]` - Level 1未匹配，Level 2相似查询复用
- `["level1_rules", "level3_llm"]` - Level 1和Level 2均未匹配，Level 3 LLM分类

**路由模式：**
- `"auto"`: 基于复杂度分析的自动路由
- `"light"`: 强制Tier 1（轻量模型）
- `"medium"`: 强制Tier 2（中端模型）
- `"glm5"`: 强制Tier 3（GLM5/高端模型）

#### 其他接口

- `GET /health` - 健康检查
- `GET /admin/stats` - 路由统计信息
- `POST /admin/config` - 更新路由配置

## 项目结构

```
llm-router/
├── src/
│   ├── router/               # Router Gateway
│   │   ├── main.py          # FastAPI应用入口
│   │   ├── api/
│   │   │   ├── completions.py    # 聊天完成API
│   │   │   └── admin.py          # 管理端点
│   │   ├── config.py        # 配置管理
│   │   └── models.py        # Pydantic模型
│   │
│   ├── classifier/           # 三级分类系统
│   │   ├── level1_rules.py       # Level 1: 基于规则的关键词分类器
│   │   ├── level2_embedding.py   # Level 2: 基于向量Embedding的语义相似度匹配
│   │   │                          #   - EmbeddingService: 文本向量化编码服务
│   │   │                          #   - VectorStore: 内存向量存储（线程安全）
│   │   │                          #   - Level2SimilarityMatcher: 相似度匹配器
│   │   ├── level3_llm.py         # Level 3: 基于LLM的复杂度分类器
│   │   └── router.py             # 路由编排器（整合三级分类）
│   │
│   ├── models/               # 模型池管理
│   │   ├── pool.py              # 模型实例池
│   │   ├── load_balancer.py     # 负载均衡策略
│   │   ├── glm5_client.py       # GLM5客户端
│   │   └── lightweight_client.py # 轻量模型客户端
│   │
│   └── common/               # 公共工具
│       └── logger.py            # 日志配置
│
├── tests/                    # 测试套件
│   └── test_classifier/
│       ├── test_level1_rules.py     # Level 1分类器测试
│       ├── test_level2_embedding.py # Level 2语义匹配测试（新增）
│       └── ...
├── config/                   # 配置文件
│   └── rules.yaml               # Level 1路由规则
├── scripts/                  # 部署脚本
│   └── deploy/
│       └── start.sh             # 启动脚本
└── docs/                     # 文档
    ├── DEPLOYMENT.md            # 详细部署指南
    └── superpowers/specs/       # 设计规范
        └── 2026-03-26-llm-router-design.md
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

## 配置说明

### 环境变量

```bash
# Level 2 配置（可选）
ENABLE_LEVEL2=true                    # 启用/禁用Level 2语义匹配（默认: true）
LEVEL2_SIMILARITY_THRESHOLD=0.85     # 余弦相似度阈值（默认: 0.85）
LEVEL2_TOP_K=5                        # 检查的最相似查询数量（默认: 5）
LEVEL2_MAX_STORE_SIZE=100000         # 向量存储最大容量（默认: 100000）

# 模型配置
GLM5_BASE_URL=http://your-glm5-server:8000
LIGHTWEIGHT_BASE_URLS=http://qwen-001:8000,http://qwen-002:8000
LIGHTWEIGHT_MODEL_NAME=qwen2.5-7b

# 路由配置
ROUTER_PORT=8000
ROUTER_LOG_LEVEL=INFO
```

### 路由规则

编辑 `config/rules.yaml` 配置Level 1基于规则的关键词匹配：

```yaml
# 简单查询关键词
simple_keywords:
  - "hello"
  - "hi"
  - "你好"
  - "什么是"
  - "解释"

# 复杂查询关键词
complex_keywords:
  - "code"
  - "代码"
  - "algorithm"
  - "算法"
  - "analyze"
  - "分析"
  - "design"
  - "设计"

# 分类阈值
thresholds:
  tier1: 0.3    # 低于此值 -> Tier 1
  tier2: 0.7    # 高于此值 -> Tier 3

token_count:
  simple_max: 100
  complex_min: 2000
```

## 技术实现

### Level 2: 语义匹配架构

Level 2分类器使用 **sentence-transformers** 进行基于向量Embedding的语义相似度匹配：

**核心组件：**

1. **EmbeddingService** (`src/classifier/level2_embedding.py`)
   - 模型: `all-MiniLM-L6-v2` (384维向量)
   - 懒加载: 首次使用时加载模型，减少启动时间
   - 支持批量编码提升效率
   - 余弦相似度计算（带数值稳定性保护，epsilon = 1e-8）

2. **VectorStore** (`src/classifier/level2_embedding.py`)
   - 使用 `asyncio.Lock` 实现线程安全的内存存储
   - FIFO淘汰策略，超过最大容量时移除最旧记录
   - O(n) 相似度搜索（适合MVP；生产环境建议使用Milvus）
   - 存储内容: 文本、向量Embedding、元数据（路由决策、复杂度评分、置信度）

3. **Level2SimilarityMatcher** (`src/classifier/level2_embedding.py`)
   - 可配置相似度阈值（默认: 0.85）
   - 可配置top-k搜索（默认: 5）
   - 异步API兼容FastAPI
   - 优雅错误处理: 失败时返回None，自动降级到Level 3

**工作原理：**

```
1. 查询到达路由器
2. Level 1规则检查（关键词、token数量）
   ├─ 匹配简单查询 -> 路由到Tier 1 (60-70%的查询)
   └─ 匹配复杂查询 -> 路由到Tier 3 (10-15%的查询)
   
3. Level 2语义匹配（剩余15-30%的查询）
   ├─ 将查询编码为384维向量
   ├─ 在向量存储中搜索相似历史查询
   ├─ 相似度 >= 0.85 -> 复用历史决策 (加速10-30%的查询)
   └─ 无匹配 -> 继续到Level 3

4. Level 3 LLM分类 (20-30%的查询)
   └─ 使用Qwen2.5-7B分类 -> 路由到合适层级

5. 将路由决策存入Level 2向量存储
   └─ 未来相似查询将复用此决策
```

**生产环境考虑：**

- **当前**: 内存VectorStore（适合单实例部署）
- **未来**: 替换为Milvus实现分布式向量搜索
- **模型大小**: all-MiniLM-L6-v2约80MB，加载时间1-2秒
- **内存使用**: 10万条记录约占用400MB（384维float32向量）

### 线程安全

Level 2组件使用 `asyncio.Lock` 确保在异步FastAPI环境中的线程安全：

```python
async def add(self, text: str, embedding: np.ndarray, metadata: Dict):
    async with self._lock:
        # 临界区: 修改共享列表
        self._texts.append(text)
        self._embeddings.append(embedding)
        self._metadata.append(metadata)
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
