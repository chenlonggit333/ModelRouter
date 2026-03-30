# LLM智能路由层设计文档

**版本**: 1.0  
**日期**: 2026-03-26  
**状态**: 待审查  

---

## 1. 项目背景与目标

### 1.1 问题现状

- **资源瓶颈**: GLM5部署在2台H200*8卡，但4000只openclaw（即将扩展到50000只）的所有请求都直接访问GLM5
- **资源浪费**: 简单问题占用高端模型资源，复杂问题因排队导致用户体验差
- **外部模型局限**: 即便使用Kimi2.5等外部模型，也无法识别输入难易程度，简单问题仍耗费过多资源

### 1.2 项目目标

构建智能路由层，实现：
- **智能分流**: 简单问题→轻量模型（7B/14B），复杂问题→GLM5/Kimi2.5/GPT5.4
- **SLA保障**: 简单问题<10秒，复杂问题2-5分钟
- **准确率**: 路由准确率≥90%
- **成本优化**: GLM5调用量降低60-70%，整体成本降低50%+

---

## 2. 系统架构

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                        Openclaw Clients                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Router Gateway (3-5台)                     │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ 规则筛选     │ 负载均衡     │ 降级处理                 │ │
│  │ (Level 1)    │ 策略         │ (Circuit Breaker)        │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────────┐
│规则引擎  │    │Embedding │    │分类器服务    │
│(本地)    │    │匹配服务  │    │(Qwen2.5-7B)  │
│<1ms      │    │10-30ms   │    │50-100ms      │
└────┬─────┘    └────┬─────┘    └──────┬───────┘
     │               │                 │
     └───────────────┼─────────────────┘
                     ▼
        ┌────────────────────────┐
        │    Model Pool (模型池)  │
        ├────────────────────────┤
        │ Tier 1: 轻量模型(7B/14B)│ ← 中低端服务器 10-20台
        │ Tier 2: 中端模型(32B)   │ ← A100/H100 2-4台
        │ Tier 3: 高端模型(GLM5)  │ ← H200*2*8卡
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │  Feedback Collector    │
        │  (自动评估+策略优化)    │
        └────────────────────────┘
```

### 2.2 部署架构

| 组件 | 机器规格 | 数量 | QPS能力 |
|------|---------|------|---------|
| Router Gateway | 中低端服务器 | 3-5台 | 5000+ |
| 规则引擎 | Gateway本地 | - | 零延迟 |
| Embedding服务 | CPU服务器 | 2-4台 | 1000+ |
| 分类器(Qwen2.5-7B) | 中低端服务器 | 4-8台 | 400-800 |
| 轻量模型池(7B) | 中低端服务器 | 10-20台 | 1000-2000 |
| 中端模型池(32B) | A100/H100 | 2-4台 | 80-160 |
| 高端模型(GLM5) | H200*2*8卡 | 现有 | 保持现状 |

---

## 3. 智能分类器设计

### 3.1 三级路由决策机制

#### Level 1: 极速规则筛选 (<1ms)

**覆盖场景**: 60-70%的简单请求，零模型调用

```python
def level1_rules(input_text, token_count):
    # 明显简单：短文本且无复杂关键词
    if token_count < 100 and not has_keywords(input_text, ["代码", "分析", "推理", "计算"]):
        return RouteDecision.TIER1_SIMPLE
    
    # 明显复杂：长文本或明确复杂意图
    if token_count > 2000 or has_keywords(input_text, ["复杂", "详细", "深入", "完整方案"]):
        return RouteDecision.TIER3_COMPLEX
    
    # 无法判断，进入Level 2
    return RouteDecision.CONTINUE
```

**关键词库示例**:
- **简单指示词**: "你好", "谢谢", "什么是", "介绍一下"
- **复杂指示词**: "编写", "分析", "比较", "优化", "设计"

#### Level 2: Embedding语义匹配 (10-30ms)

**覆盖场景**: 历史相似问题复用

**技术方案**:
- 模型: `sentence-transformers/all-MiniLM-L6-v2` (80MB)
- 向量数据库: Milvus
- 匹配策略: Cosine Similarity > 0.85

**流程**:
1. 将输入文本转为384维embedding向量
2. 在向量数据库中查询Top-5相似历史问题
3. 如果Top-1相似度>0.85且历史路由一致 → 复用决策
4. 否则进入Level 3

**存储规划**:
- 100万条历史记录 ≈ 5GB存储
- 日增量：~5万条（基于50000 openclaw规模）
- 保留策略：最近90天数据

#### Level 3: 小模型智能分类 (50-100ms)

**覆盖场景**: 边缘情况，预计20-30%的请求

**模型配置**:
```yaml
model: Qwen2.5-7B-Instruct
framework: vLLM
deployment:
  instances: 4-8
  per_instance_batch: 16
  gpu_memory: 16GB per instance
  concurrency: 100-200 QPS total
```

**分类Prompt**:
```
你是一个问题复杂度评估专家。请评估以下问题的复杂度：

问题：{user_question}

请从以下维度评估（0-10分）：
1. 推理深度：是否需要多步逻辑推理
2. 知识广度：是否涉及跨领域知识
3. 创造性：是否需要生成创新内容
4. 精确性：是否要求严格的准确性和完整性

请以JSON格式输出：
{
  "complexity_score": 0.0-1.0,  // 综合复杂度评分
  "confidence": 0.0-1.0,        // 置信度
  "reasoning": "string",        // 主要理由（50字以内）
  "dimensions": {               // 各维度详细评分
    "reasoning_depth": 0-10,
    "knowledge_breadth": 0-10,
    "creativity": 0-10,
    "accuracy_requirement": 0-10
  }
}
```

**输出格式验证**:
- 必须返回合法JSON，否则视为分类失败
- complexity_score和confidence为浮点数，范围[0.0, 1.0]
- Gateway层解析JSON，提取complexity_score用于路由决策

### 3.2 分类阈值策略

| 复杂度评分 | 路由目标 | 代表场景 | 占比预估 |
|-----------|---------|---------|---------|
| 0.0-0.3 | Tier 1: 轻量模型 | 闲聊、简单问答、事实查询 | 60-70% |
| 0.3-0.7 | Tier 2: 中端模型 | 中等分析、代码片段、文档总结 | 15-20% |
| 0.7-1.0 | Tier 3: 高端模型 | 复杂推理、长代码、深度分析 | 10-15% |

**动态调整机制**:

**阶段1 (MVP阶段) - 手动调整**:
- 对应实施路线图：阶段1 (MVP验证)
- 初始阈值固定：tier1=0.3, tier2=0.7
- 每日生成路由质量报告
- 运维人员根据报告手动调整阈值
- 调整步长：±0.05

**阶段2 (智能增强阶段) - 自动调整**:
- **调整算法**：基于历史准确率的简单反馈控制
```python
def auto_adjust_thresholds(daily_stats):
    """
    每日自动调整阈值
    目标：维持整体准确率 >= 90%
    """
    if daily_stats.accuracy < 0.90:
        # 准确率不达标，放宽阈值，让更多请求走高端模型
        tier1_threshold = max(0.15, current_tier1 - 0.05)
        tier2_threshold = max(0.50, current_tier2 - 0.05)
        log_adjustment("放宽阈值以提升准确率")
    elif daily_stats.accuracy > 0.95 and daily_stats.tier3_ratio > 0.20:
        # 准确率很高且高端模型使用过多，收紧阈值
        tier1_threshold = min(0.40, current_tier1 + 0.03)
        tier2_threshold = min(0.75, current_tier2 + 0.03)
        log_adjustment("收紧阈值以优化成本")
    # 否则保持当前阈值
```
- **约束条件**：
  - tier1_threshold ∈ [0.15, 0.40]
  - tier2_threshold ∈ [0.50, 0.80]
  - tier1_threshold + 0.20 <= tier2_threshold（保持层级间隔）
- **人工干预**：可随时通过管理接口覆盖自动调整

---

## 4. 模型池与负载均衡

### 4.1 分层模型池

#### Tier 1: 轻量模型池

**模型选型**:
- 主模型: Qwen2.5-7B-Instruct
- 备选: ChatGLM3-6B, Baichuan2-7B

**部署配置**:
```yaml
servers: 10-20台（中低端算力）
per_server:
  instances: 2-4
  batch_size: 16
  max_concurrency: 50-100
qps_total: 1000-2000
```

**适用场景**:
- Token数 < 500
- 简单问答、闲聊
- 事实查询、定义解释

#### Tier 2: 中端模型池

**模型选型**:
- Qwen2.5-32B-Instruct
- 或其他32B级别模型

**部署配置**:
```yaml
servers: 2-4台（A100/H100）
per_server:
  instances: 1-2
  tensor_parallel: 2-4
qps_total: 80-160
```

**适用场景**:
- 中等复杂度分析
- 代码生成（<200行）
- 中等长度文档处理

#### Tier 3: 高端模型池

**模型配置**:
- GLM5 (现有H200*2*8卡)
- Kimi2.5 (外部API)
- GPT5.4 (外部API)

**适用场景**:
- 复杂推理任务
- 长代码生成/审查
- 深度文档分析

### 4.2 负载均衡策略

**基础算法**:
1. **加权轮询**: 根据实例容量分配权重
2. **最少连接**: 优先分配给当前负载最低的实例

**增强策略**:
```python
def smart_routing(tier, request):
    candidates = get_healthy_instances(tier)
    
    # 策略1: 队列深度感知
    # queue_depth定义：当前实例等待队列中的请求数
    # 阈值：如果queue_depth > 50，视为过载
    candidates = sorted(candidates, key=lambda x: x.queue_depth)
    
    # 策略2: 预估耗时调度
    # 预估处理时间 = 输入token数 / 实例吞吐量 + 排队时间
    estimated_time = estimate_processing_time(request)
    best = select_best_for_sla(candidates, estimated_time)
    
    return best

# 预估处理时间算法
def estimate_processing_time(request, instance):
    """
    预估处理时间（毫秒）
    """
    input_tokens = count_tokens(request.messages)
    processing_speed = instance.throughput_tokens_per_sec  # 实例处理能力
    queue_wait_time = instance.queue_depth * instance.avg_request_duration_ms
    
    # 处理时间 = 队列等待 + 实际处理
    estimated_processing = (input_tokens / processing_speed) * 1000
    return queue_wait_time + estimated_processing
```

**策略说明**:
- 队列深度阈值：单个实例queue_depth > 50时触发负载转移
- SLA目标：简单问题<10秒，复杂问题2-5分钟
- 选择逻辑：在满足SLA的前提下，优先选择queue_depth最小的实例

### 4.3 故障转移机制

**降级策略**:
```python
if tier1_all_busy():
    # 溢出到Tier 2
    route_to_tier2()
    log_overflow("tier1_to_tier2", request)

if tier1_instance_unhealthy(instance_id):
    # 剔除故障节点
    mark_unhealthy(instance_id)
    trigger_alert(instance_id)
    # 30秒后自动重试恢复
    schedule_health_check(instance_id, delay=30)

if classifier_service_down():
    # 分类器故障时，所有请求走GLM5
    return RouteDecision.TIER3_COMPLEX
    log_degradation("classifier_down")
```

---

## 5. 自动反馈评估机制

### 5.1 评估维度

#### 维度1: 回答充分性评分（核心指标）

**评估方案**:
- 使用轻量级模型（Qwen2.5-7B）作为"评判者"
- 异步评估，延迟1-5分钟可接受

**评分Prompt**:
```
请评估以下回答是否充分解决了用户问题：

问题：{question}
回答：{answer}

评分标准（0-10分）：
- 10分：完美解决，无需补充
- 7-9分：基本解决，有少量可改进空间
- 4-6分：部分解决，缺少关键信息
- 0-3分：未解决或答非所问

评分：___
理由：___
```

**判定规则**:
- 8分以上 → 路由成功
- 5-7分 → 可接受，但有优化空间
- <5分 → 可能的路由错误，需要分析

#### 维度2: 用户行为信号（间接指标）

**采集指标**:
```python
behavior_signals = {
    "follow_up_within_30s": bool,      # 快速追问 → 可能不充分
    "conversation_duration_minutes": float,  # 简单问题>5分钟 → 需升级
    "answer_copy_count": int,          # 复制次数多 → 有价值
    "answer_length_ratio": float,      # 回答长度/问题长度
}
```

**启发规则**:
- 简单路由问题 + 30秒内追问 → 可能误判为简单
- 复杂路由问题 + 长时间会话 → 复杂模型被正确使用

#### 维度3: 一致性检查（抽样）

**抽样策略**:
- 每日随机抽取1000条请求
- 用同一问题分别查询轻量模型和GLM5
- 对比质量差异

**判定规则**:
```python
if light_score < 5 and glm_score > 8:
    # 轻量模型表现差，GLM5表现好 → 路由错误
    mark_routing_error(request_id, "under_route")
elif light_score > 8 and glm_score > 8:
    # 两者都好 → 正确路由到轻量模型，节省成本
    mark_routing_success(request_id, "cost_saved")
```

### 5.2 反馈闭环流程

```
请求完成
    │
    ▼
[实时反馈] ─────────────────────────────┐
    │                                    │
    ▼                                    │
监控延迟/错误率                              │
    │                                    │
    ▼                                    │
触发告警（若异常）                            │
    │                                    │
    ▼                                    │
[异步反馈] ◄─────────────────────────────┘
    │
    ▼
延迟5-10分钟后深度评估
    │
    ▼
聚合评分数据
    │
    ▼
生成每日路由质量报告
    │
    ▼
自动调整分类阈值（如果准确率<90%）
    │
    ▼
每周模型重训练（增量更新）
```

### 5.3 模型迭代机制

**说明**：本节所述"模型迭代"特指**分类器模型（Qwen2.5-7B）**的优化，不涉及Tier 1/2/3业务模型的变更。

**训练数据生成**:
- 每日自动生成5000-10000条标记数据
- 数据来源：
  - 明确路由成功/失败的案例（高置信度）
  - 人工审核的边界案例
  - 一致性检查的抽样对比结果
- 数据格式：
```json
{
  "question": "用户原始问题",
  "complexity_score": 0.65,
  "route_decision": "tier2",
  "evaluation_result": "success",
  "timestamp": "2026-03-26T10:00:00Z"
}
```
- 数据清洗：去除噪声样本（置信度<0.7的分类结果）

**模型微调（Fine-tuning）**:
- **触发条件**：
  - 连续3天整体准确率<85%，且阈值调整无效
  - 累积高质量标记数据>5万条
- **微调范围**：
  - 基于Qwen2.5-7B-Instruct进行LoRA微调
  - 学习率：1e-4，epochs：3
  - 仅更新分类器头部参数，保持基础模型不变
- **评估标准**：
  - 验证集准确率提升>5%才接受新模型
  - A/B测试：新模型部署到10%流量，对比旧模型

**轻量级更新（无需微调）**:
- **Embedding库更新**（每周）：
  - 增量添加新的历史问题向量
  - 清理过期数据（>90天）
  - 重建索引以优化查询性能
  
- **阈值调整**（每日）：
  - 参见3.2节的动态调整机制
  - 纯配置更新，无需模型重训练

**版本管理**:
- 每个微调版本保存checkpoint
- 支持快速回滚到上一版本
- 保留最近3个历史版本

---

## 6. 技术选型

### 6.1 核心服务

| 组件 | 技术选型 | 理由 |
|------|---------|------|
| Router Gateway | Python (FastAPI) | 团队熟悉，开发快，生态丰富 |
| 分类器推理 | vLLM | 高性能推理，支持并发 |
| Embedding模型 | sentence-transformers | 轻量，CPU可跑 |
| 向量数据库 | Milvus | 开源，高性能，易扩展 |
| 时序数据库 | InfluxDB | 适合metrics存储 |
| 缓存/配置 | Redis | 高性能KV存储 |
| 持久化存储 | PostgreSQL | 关系型数据存储 |

### 6.2 模型选型

| 层级 | 模型 | 部署位置 | 显存需求 |
|------|------|---------|---------|
| 分类器 | Qwen2.5-7B-Instruct | 中低端服务器 | 16GB |
| Tier 1 | Qwen2.5-7B-Instruct | 中低端服务器 | 16GB |
| Tier 2 | Qwen2.5-32B-Instruct | A100/H100 | 80GB |
| Tier 3 | GLM5 | H200*2*8卡 | 现有配置 |

---

## 7. 接口设计

### 7.1 对外接口（兼容OpenAI格式）

**路由请求**:
```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <token>
X-Router-Debug: true  # 可选，返回路由详情

{
  "model": "auto",  # 或指定 "glm5", "light", "medium"
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000
}
```

**响应示例**:
```json
{
  "id": "chatcmpl-abc123",
  "model": "qwen2.5-7b",
  "created": 1677652288,
  "router_info": {
    "complexity_score": 0.15,
    "route_decision": "tier1",
    "classification_time_ms": 45,
    "routing_path": ["level1_rules", "level2_embedding"]
  },
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### 7.2 管理接口

**查看统计**:
```http
GET /admin/stats?time_range=24h&granularity=hour

{
  "total_requests": 100000,
  "routing_distribution": {
    "tier1": 65000,
    "tier2": 20000,
    "tier3": 15000
  },
  "avg_latency_ms": {
    "classification": 35,
    "tier1": 800,
    "tier2": 3000,
    "tier3": 8000
  },
  "accuracy": 0.92,
  "error_rate": 0.001
}
```

**调整配置**:
```http
POST /admin/config
{
  "tier1_threshold": 0.3,
  "tier2_threshold": 0.7,
  "enable_embedding": true,
  "enable_classifier": true
}
```

**健康检查**:
```http
GET /admin/health

{
  "status": "healthy",
  "components": {
    "gateway": "up",
    "classifier": "up",
    "tier1_pool": "up",
    "tier2_pool": "up",
    "tier3_pool": "up"
  },
  "metrics": {
    "qps": 1500,
    "queue_depth": 12
  }
}
```

---

## 8. 实施路线图

### 阶段1: MVP验证（2-3周）

**目标**: 快速验证核心假设，达到70%路由准确率

**Week 1**:
- [ ] 部署Router Gateway基础框架（FastAPI）
- [ ] 实现Level 1规则筛选（覆盖50%简单请求）
- [ ] 对接GLM5（现有）和1个轻量模型（Qwen2.5-7B）
- [ ] 基础监控和日志

**Week 2-3**:
- [ ] 部署小模型分类器（4-8实例）
- [ ] 实现基础负载均衡（轮询+最少连接）
- [ ] 接入10%真实流量进行A/B测试
- [ ] 收集初步反馈数据

**成功标准**:
- 路由延迟（分类部分）< 100ms
- 简单模型分流比例 > 40%
- 无重大故障，可用性>99%

### 阶段2: 智能增强（3-4周）

**目标**: 提升至90%准确率，完整反馈闭环

**Week 4-5**:
- [ ] 部署Embedding服务和Milvus向量数据库
- [ ] 接入Level 2语义匹配
- [ ] 实现Embedding相似度路由

**Week 6-7**:
- [ ] 部署Feedback Collector服务
- [ ] 实现自动评估（充分性评分）
- [ ] 实现自动阈值调整机制

**Week 8**:
- [ ] 全量切换（按业务要求）
- [ ] 监控大盘和告警系统
- [ ] 性能调优和容量评估

**成功标准**:
- 路由准确率 ≥ 90%
- GLM5调用量降低 50%+
- 简单问题平均响应时间 < 5秒

### 阶段3: 规模优化（持续迭代）

**目标**: 支持50000只openclaw，成本最优

**优化方向**:
- [ ] 多轻量模型组合（针对不同领域优化）
- [ ] 模型蒸馏：用GLM5生成数据训练专属轻量模型
- [ ] 动态扩缩容（基于负载自动调整实例数）
- [ ] A/B测试框架（持续优化路由策略）

**预期收益**:
- GLM5调用量降低 60-70%
- 平均成本降低 50%+
- 简单问题响应速度提升 5-10倍
- 支持 50000+ openclaw并发

---

## 9. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| 路由准确率不达标 | 高 | 保留降级开关，可随时切回纯GLM5 |
| 分类器延迟过高 | 中 | 规则筛选覆盖大部分请求，仅边缘情况走分类器 |
| 轻量模型过载 | 中 | 自动溢出到下一层级，保证可用性 |
| 向量数据库瓶颈 | 低 | 可横向扩展，且仅影响Level 2 |
| 数据隐私问题 | 低 | 日志脱敏处理，敏感数据不入库 |

---

## 10. 成功指标

### 10.1 技术指标

| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| 路由准确率 | ≥90% | 自动评估+人工抽样 |
| 分类延迟 | <100ms | Gateway日志统计 |
| 系统可用性 | ≥99.9% | 健康检查 |
| 降级响应时间 | <1秒 | 压力测试 |

### 10.2 业务指标

| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| GLM5调用量降低 | 60-70% | 接口调用统计 |
| 整体成本降低 | 50%+ | 资源使用统计 |
| 简单问题响应时间 | <5秒 | 端到端延迟 |
| 用户满意度 | 无显著下降 | 业务反馈 |

---

## 11. 附录

### 11.1 术语表

- **openclaw**: 公司内部调用端（共50000只）
- **GLM5**: 内部部署的高端大模型
- **Tier 1/2/3**: 轻量/中端/高端模型层级
- **Embedding**: 文本向量表示
- **Qwen2.5-7B**: 轻量级中文大模型

### 11.2 参考资料

- vLLM文档: https://docs.vllm.ai/
- Milvus文档: https://milvus.io/docs
- FastAPI文档: https://fastapi.tiangolo.com/
- Qwen2.5模型卡: https://huggingface.co/Qwen

---

**文档维护**:
- 作者: AI Assistant
- 最后更新: 2026-03-26
- 下次审查: 实施阶段1完成后
