# LLM智能路由层 (LLM Router)

智能路由层，实现请求分发到不同层级的LLM模型，优化成本和响应速度。

## 功能特性

- **智能路由**: 根据问题复杂度自动选择合适模型（轻量/中端/高端）
- **三级分类**: Level 1规则筛选 + Level 3 LLM分类
- **负载均衡**: 支持轮询、最少连接等多种策略
- **降级机制**: 实例故障时自动切换，保证可用性
- **OpenAI兼容API**: 完全兼容OpenAI接口格式

## 架构

```
用户请求 → Router Gateway → [Level 1规则筛选] → [Level 3 LLM分类] → 模型池 → 响应
                                    ↓                        ↓
                            60-70%简单请求            30-40%复杂请求
                                    ↓                        ↓
                            轻量模型(7B)              GLM5/H100
```

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
│   ├── classifier/      # 分类器（Level 1 + Level 3）
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

### 生产环境部署

1. 准备配置文件 `config/rules.yaml`
2. 设置环境变量 `.env`
3. 启动服务: `./scripts/deploy/start.sh`
4. 配置Nginx反向代理
5. 配置监控告警

### Docker部署（可选）

```bash
docker build -t llm-router .
docker run -p 8000:8000 --env-file .env llm-router
```

## 许可证

MIT
