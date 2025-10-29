# 如何重写 SQL Agent 的四个工具

## 概述

LangChain SQL Agent 默认有四个工具：
1. `sql_db_list_tables` - 列出所有表
2. `sql_db_schema` - 获取表结构
3. `sql_db_query` - 执行 SQL 查询
4. `sql_db_query_checker` - 检查 SQL 语法

本文档展示如何自定义这些工具，添加额外的功能，如：
- 详细的日志记录
- SQL 注入防护
- 查询结果限制
- 执行时间统计
- 强制执行流程检查

## 核心概念

### 1. 工具的基本结构

每个自定义工具需要继承 `BaseTool` 并实现：

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class CustomTool(BaseTool):
    # 工具名称（Agent 通过这个名称调用）
    name = "tool_name"
    
    # 工具描述（告诉 LLM 这个工具的作用和如何使用）
    description = """工具的详细描述..."""
    
    # 输入参数的 schema
    args_schema: Type[BaseModel] = InputSchema
    
    # 工具需要的额外字段（如数据库连接）
    db: SQLDatabase = Field(exclude=True)
    
    # 核心方法：执行工具的逻辑
    def _run(self, input_param: str) -> str:
        # 工具的具体实现
        pass
```

### 2. 关键改进点

#### 工具 1: sql_db_list_tables
**改进方向：**
- ✅ 添加表名过滤（隐藏系统表）
- ✅ 记录访问日志
- ✅ 添加表的元信息（如行数）

#### 工具 2: sql_db_schema  
**改进方向：**
- ✅ 验证表是否存在
- ✅ 控制示例行数
- ✅ 格式化输出
- ✅ 添加索引信息

#### 工具 3: sql_db_query（最重要）
**改进方向：**
- ✅ **SQL 注入防护**（禁止 DML 语句）
- ✅ **自动添加 LIMIT**（防止返回过多数据）
- ✅ **详细的执行日志**
- ✅ **统计查询性能**
- ✅ **追踪调用历史**
- ✅ **在描述中强调必须先检查**

#### 工具 4: sql_db_query_checker
**改进方向：**
- ✅ 基本语法检查
- ✅ 安全检查（禁止危险操作）
- ✅ **在返回结果中提示下一步必须执行查询**
- ✅ 可选的 LLM 智能检查

## 使用方法

### 方法 1: 直接使用自定义工具包

```python
from sql_agent.custom_sql_tools import CustomSQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

# 1. 创建数据库连接
db = SQLDatabase.from_uri("sqlite:///company.db")

# 2. 创建自定义工具包
toolkit = CustomSQLDatabaseToolkit(
    db=db,
    sample_rows=3,      # schema 工具返回的示例行数
    max_results=100     # query 工具最大返回行数
)

# 3. 获取工具列表
tools = toolkit.get_tools()

# 4. 在 Agent 中使用
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)
```

### 方法 2: 在现有 Agent 中替换工具

```python
from sql_agent.custom_sql_tools import (
    CustomSQLDatabaseQuery,
    CustomSQLDatabaseQueryChecker
)

# 只替换关键的两个工具
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# 获取默认工具
default_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = default_toolkit.get_tools()

# 找到并替换 query 和 checker 工具
for i, tool in enumerate(tools):
    if tool.name == "sql_db_query":
        tools[i] = CustomSQLDatabaseQuery(db=db, max_results=50)
    elif tool.name == "sql_db_query_checker":
        tools[i] = CustomSQLDatabaseQueryChecker(db=db, llm=llm)

# 使用替换后的工具创建 Agent
agent = create_sql_agent(llm, db, tools=tools)
```

### 方法 3: 创建完全自定义的工具

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class MyCustomQueryTool(BaseTool):
    name = "sql_db_query"
    description = """自定义描述，强调执行流程"""
    
    db: SQLDatabase = Field(exclude=True)
    
    def _run(self, query: str) -> str:
        # 1. 记录到专门的日志文件
        with open("query_log.txt", "a") as f:
            f.write(f"{datetime.now()}: {query}\n")
        
        # 2. 添加自定义验证
        if not self._validate_query(query):
            return "查询被拒绝"
        
        # 3. 执行查询
        result = self.db.run(query)
        
        # 4. 后处理结果
        return self._format_result(result)
    
    def _validate_query(self, query: str) -> bool:
        # 自定义验证逻辑
        pass
    
    def _format_result(self, result: str) -> str:
        # 自定义格式化逻辑
        pass
```

## 完整示例

查看 `custom_sql_tools.py` 中的 `demo_custom_tools()` 函数：

```bash
# 运行演示
python -m sql_agent.custom_sql_tools
```

## 核心改进：解决"跳步"问题

### 问题
模型经常跳过 `sql_db_query` 步骤，直接给出编造的答案。

### 解决方案
在自定义工具中：

1. **在 `sql_db_query_checker` 的返回结果中添加强制提示：**
```python
result = f"{checked_query}\n\n✅ 语法检查通过！\n⚠️ 下一步：请使用 sql_db_query 执行此查询以获取实际结果。"
```

2. **在 `sql_db_query` 的描述中强调其重要性：**
```python
description = """
...
⚠️ 重要：你**必须**先使用 sql_db_query_checker 检查 SQL 语法，然后才能使用此工具！
这是执行实际查询的工具，会返回真实的数据库结果。
"""
```

3. **添加执行统计，监控是否真的执行了查询：**
```python
query_tool.get_statistics()
# 输出: {'total_calls': 3, 'successful': 3, ...}
```

## 高级技巧

### 1. 添加查询缓存

```python
class CachedSQLDatabaseQuery(CustomSQLDatabaseQuery):
    cache: dict = Field(default_factory=dict)
    
    def _run(self, query: str) -> str:
        # 检查缓存
        if query in self.cache:
            logger.info("💾 使用缓存结果")
            return self.cache[query]
        
        # 执行查询
        result = super()._run(query)
        
        # 存入缓存
        self.cache[query] = result
        return result
```

### 2. 添加查询审计

```python
class AuditedSQLDatabaseQuery(CustomSQLDatabaseQuery):
    def _run(self, query: str) -> str:
        # 保存到审计日志
        self._save_to_audit_log({
            'timestamp': datetime.now(),
            'query': query,
            'user': 'system',
        })
        
        return super()._run(query)
```

### 3. 添加查询优化建议

```python
class SmartSQLDatabaseQueryChecker(CustomSQLDatabaseQueryChecker):
    def _run(self, query: str) -> str:
        # 基本检查
        result = super()._run(query)
        
        # 使用 LLM 提供优化建议
        if self.llm:
            suggestions = self._get_optimization_suggestions(query)
            result += f"\n\n💡 优化建议：\n{suggestions}"
        
        return result
```

## 调试技巧

### 查看工具调用历史

```python
# 获取 query 工具的调用历史
query_tool = [t for t in tools if t.name == "sql_db_query"][0]
history = query_tool.call_history

for call in history:
    print(f"时间: {call['timestamp']}")
    print(f"查询: {call['query']}")
    print(f"状态: {call['status']}")
    print(f"耗时: {call.get('execution_time', 'N/A')}s")
    print("-" * 80)
```

### 监控工具使用情况

```python
# 统计每个工具被调用的次数
from collections import Counter

tool_calls = []
# 在 callback 中收集工具调用记录

counter = Counter(tool_calls)
print("工具使用统计:")
for tool_name, count in counter.items():
    print(f"  {tool_name}: {count} 次")
```

## 常见问题

### Q1: 自定义工具不生效？
**A**: 确保在创建 Agent 时传入了自定义工具：
```python
agent = create_sql_agent(llm, db, tools=custom_tools)  # ✅
# 而不是
agent = create_sql_agent(llm, db)  # ❌ 使用默认工具
```

### Q2: 如何验证工具是否被正确加载？
**A**: 打印工具列表：
```python
for tool in tools:
    print(f"工具: {tool.name}")
    print(f"类型: {type(tool).__name__}")
```

### Q3: 模型仍然跳过工具调用？
**A**: 可能需要：
1. 增强 Prompt 中的流程说明
2. 使用 Few-Shot 示例
3. 使用更强的模型（GPT-4 而非 Qwen）
4. 在代码层面强制检查工具调用顺序

## 参考资料

- LangChain 官方文档: https://python.langchain.com/docs/modules/agents/tools/
- SQL Agent 源码: `langchain_community.agent_toolkits.sql`
- 本项目示例: `sql_agent/custom_sql_tools.py`

