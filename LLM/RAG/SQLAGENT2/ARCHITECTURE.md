# SQL Agent 架构说明

## 整体架构

SQL Agent采用模块化设计，每个模块负责特定的功能，通过清晰的接口协作完成复杂的数据查询任务。

## 核心组件

### 1. Agent (`agent.py`)

**职责**: 核心协调器，整合所有模块，实现完整的Agent逻辑流程

**主要功能**:
- 接收用户查询
- 协调各个模块执行任务
- 处理不同类型的查询（简单/复杂/澄清）
- 管理执行状态和错误处理

**关键方法**:
```python
query(question: str) -> str  # 主入口，处理用户查询
_handle_simple_query()        # 处理简单查询
_handle_complex_query()       # 处理复杂查询
_handle_clarification()       # 处理澄清需求
```

### 2. 意图路由器 (`intent_router.py`)

**职责**: 识别用户问题的类型并路由到相应的处理流程

**分类类型**:
- `simple_query`: 简单查询，可直接生成SQL
- `complex_query`: 复杂查询，需要多步骤推理
- `clarification_needed`: 需要澄清，问题不够明确

**工作流程**:
```
用户问题 → LLM分析 → 返回分类结果
                    (意图类型 + 理由 + 置信度)
```

### 3. SQL生成器 (`sql_generator.py`)

**职责**: 将自然语言问题转换为SQL查询

**特性**:
- 基于数据库schema生成准确的SQL
- 支持Few-shot学习（使用历史成功查询作为示例）
- 支持SQL修正（执行失败时自动修正）

**输入**:
- 自然语言问题
- 数据库schema
- 上下文信息（业务规则、查询模式）
- 示例查询（可选）

**输出**:
- SQL查询语句
- 查询解释
- 置信度

### 4. 查询执行器 (`query_executor.py`)

**职责**: 执行SQL查询并格式化结果

**功能**:
- 安全地执行SQL查询
- 将结果转换为DataFrame
- 支持多种输出格式（自然语言/表格/JSON）
- 提供结果统计摘要

**格式化选项**:
```python
format_results(df, format_type="natural")  # 自然语言描述
format_results(df, format_type="table")    # 表格格式
format_results(df, format_type="json")     # JSON格式
```

### 5. 规划器 (`planner.py`)

**职责**: 将复杂问题分解为多个子任务并规划执行顺序

**任务类型**:
- `sql_query`: SQL查询任务
- `calculation`: 计算任务
- `aggregation`: 聚合任务

**任务依赖管理**:
- 识别任务之间的依赖关系
- 按照依赖顺序执行任务
- 传递中间结果给依赖任务

**执行流程**:
```
复杂问题 → 任务分解 → 创建执行计划
         → 循环执行子任务 → 汇总结果
```

### 6. 澄清器 (`clarifier.py`)

**职责**: 当问题不明确时，与用户交互获取更多信息

**功能**:
- 识别问题中的模糊或缺失信息
- 生成友好的澄清问题
- 处理用户回答，生成完整问题

**交互流程**:
```
模糊问题 → 生成澄清问题 → 用户回答 
        → 组合完整问题 → 重新处理
```

## 支持模块

### 7. 知识库 (`knowledge_base.py`)

**职责**: 存储和管理数据库相关知识

**存储内容**:
- 数据库Schema（表结构、字段类型、主外键）
- 表描述和列描述
- 业务规则
- 常见查询模式

**使用场景**:
- SQL生成时提供上下文
- 意图识别时理解业务语义
- 查询优化时应用业务规则

### 8. 安全模块 (`security.py`)

**职责**: 确保SQL查询的安全性

**安全级别**:
- `LOW`: 只禁止最危险的操作（DROP、SHUTDOWN）
- `MEDIUM`: 禁止大部分修改操作（DROP、ALTER、CREATE等）
- `HIGH`: 只允许SELECT查询

**安全检查**:
```python
1. 检查禁止的SQL关键字
2. 检测SQL注入模式
3. 验证表访问权限
4. 添加LIMIT限制
```

### 9. 记忆系统 (`memory.py`)

**职责**: 存储和检索历史信息

**记忆类型**:
- **对话记忆**: 存储用户和Agent的对话历史
- **执行记忆**: 记录任务执行历史
- **SQL记忆**: 保存SQL查询历史（成功和失败）
- **知识记忆**: 存储用户偏好、反馈等

**应用**:
- 提供上下文理解
- Few-shot学习（使用历史成功查询）
- 避免重复错误
- 个性化回答

### 10. 配置管理 (`config.py`)

**职责**: 管理应用配置

**配置项**:
- OpenAI API配置（密钥、模型、温度）
- 数据库连接配置
- Agent行为配置（最大迭代次数、详细输出）

## 数据流图

### 简单查询流程

```
用户问题
    ↓
意图识别 → simple_query
    ↓
SQL生成 ← 知识库提供schema和上下文
    ↓      ← 记忆系统提供成功查询示例
安全检查
    ↓
查询执行
    ↓
结果格式化
    ↓
返回用户
```

### 复杂查询流程

```
用户问题
    ↓
意图识别 → complex_query
    ↓
任务规划 ← 知识库提供schema
    ↓
循环执行子任务:
  ├─ 子任务1 → SQL生成 → 执行 → 结果1
  ├─ 子任务2 → SQL生成 → 执行 → 结果2
  │            ↑
  │            └─ 使用结果1作为上下文
  └─ 子任务n → SQL生成 → 执行 → 结果n
    ↓
汇总所有结果
    ↓
生成最终回答
    ↓
返回用户
```

### 澄清流程

```
用户问题
    ↓
意图识别 → clarification_needed
    ↓
生成澄清问题 ← 知识库提供schema和业务规则
    ↓
返回澄清问题
    ↓
用户回答
    ↓
组合完整问题
    ↓
重新进入查询流程
```

## 模块依赖关系

```
┌─────────────┐
│    Agent    │ (核心协调器)
└──────┬──────┘
       │
       ├──────────────────┬──────────────────┬──────────────────┐
       │                  │                  │                  │
┌──────▼──────┐    ┌─────▼──────┐    ┌─────▼──────┐    ┌─────▼──────┐
│IntentRouter │    │SQLGenerator│    │   Planner  │    │ Clarifier  │
└─────────────┘    └─────┬──────┘    └─────┬──────┘    └────────────┘
                         │                  │
                   ┌─────▼──────┐    ┌──────▼──────┐
                   │QueryExecutor│    │             │
                   └─────────────┘    │             │
                                     │             │
       ┌──────────────────────────────┴─────────────┴───┐
       │                                                  │
┌──────▼──────┐    ┌──────────┐    ┌──────────┐
│KnowledgeBase│    │ Security │    │  Memory  │
└─────────────┘    └──────────┘    └──────────┘
```

## 扩展点

### 1. 添加新的任务类型

在 `planner.py` 中定义新的任务类型，在 `agent.py` 中实现处理逻辑：

```python
# planner.py
class SubTask(BaseModel):
    query_type: str  # 添加新类型，如 "api_call", "file_operation"
    
# agent.py
def _execute_task(self, task):
    if task.query_type == "sql_query":
        return self._execute_sql_task(task)
    elif task.query_type == "api_call":
        return self._execute_api_task(task)  # 新增
```

### 2. 支持新的数据源

继承或扩展 `KnowledgeBase` 类：

```python
class MongoKnowledgeBase(KnowledgeBase):
    def __init__(self, mongo_uri):
        # 实现MongoDB连接
        pass
    
    def get_database_schema(self):
        # 实现MongoDB schema获取
        pass
```

### 3. 自定义输出格式

在 `QueryExecutor` 中添加新的格式化方法：

```python
def format_results(self, df, format_type="natural"):
    if format_type == "markdown":
        return self._format_as_markdown(df)  # 新增
    # ...
```

### 4. 集成其他LLM

替换 `ChatOpenAI` 为其他LLM实现：

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3")
agent = SQLAgent(llm=llm, ...)
```

## 性能优化

### 1. 缓存机制

可以添加查询结果缓存：

```python
class CachedQueryExecutor(QueryExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
    
    def execute_query(self, sql):
        if sql in self.cache:
            return self.cache[sql]
        result = super().execute_query(sql)
        self.cache[sql] = result
        return result
```

### 2. 并行执行

对于独立的子任务，可以并行执行：

```python
import asyncio

async def execute_tasks_parallel(self, tasks):
    results = await asyncio.gather(*[
        self._execute_task_async(task) 
        for task in tasks
    ])
    return results
```

### 3. Schema缓存

避免每次查询都反射数据库Schema：

```python
class KnowledgeBase:
    def __init__(self, database_uri):
        self._schema_cache = None
    
    def get_database_schema(self):
        if self._schema_cache is None:
            self._schema_cache = self._load_schema()
        return self._schema_cache
```

## 测试策略

### 单元测试
- 每个模块独立测试
- 使用Mock对象模拟依赖
- 参考 `test_agent.py`

### 集成测试
- 测试模块间的协作
- 使用测试数据库

### 端到端测试
- 测试完整的查询流程
- 使用真实数据库和OpenAI API

## 最佳实践

1. **详细的知识库**: 为表和列添加详细描述，提高SQL生成准确性
2. **合理的安全级别**: 根据使用场景选择合适的安全级别
3. **监控和日志**: 开启verbose模式，记录查询过程
4. **错误处理**: 捕获并友好地处理各种错误
5. **成本控制**: 合理使用缓存，减少LLM调用次数

## 未来改进

- [ ] 支持流式输出
- [ ] 添加查询优化建议
- [ ] 支持多轮对话上下文
- [ ] 集成可视化组件
- [ ] 支持多数据源联合查询
- [ ] 添加查询性能分析
- [ ] 实现查询结果导出功能

