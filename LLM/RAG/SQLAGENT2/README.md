# SQL Agent - 智能数据查询助手

基于LangChain框架的智能SQL Agent，能够将自然语言问题转换为SQL查询，支持复杂多步推理、智能澄清等功能。

## 功能特性

### 核心功能

1. **意图识别与路由**
   - 自动识别问题类型（简单查询/复杂查询/需要澄清）
   - 智能路由到相应的处理流程

2. **简单查询处理**
   - 直接生成并执行SQL
   - 自动添加安全限制
   - 失败时自动修正重试

3. **复杂多步查询**
   - 自动分解为子任务
   - 识别任务依赖关系
   - 按序执行并汇总结果

4. **交互式澄清**
   - 识别模糊问题
   - 生成友好的澄清问题
   - 处理用户回答并继续执行

### 辅助模块

- **知识库**: 存储数据库schema、业务规则、查询模式
- **安全模块**: SQL注入检测、权限控制、查询限制
- **记忆系统**: 对话历史、执行历史、知识记忆

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      SQL Agent                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │ 意图识别路由  │─────>│  问题分类     │               │
│  └──────────────┘      └──────┬───────┘               │
│                               │                         │
│         ┌─────────────────────┼─────────────────────┐  │
│         │                     │                     │  │
│         ▼                     ▼                     ▼  │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐│
│  │简单查询   │         │复杂查询   │         │需要澄清   ││
│  │处理器    │         │处理器    │         │处理器    ││
│  └──────────┘         └──────────┘         └──────────┘│
│         │                     │                     │  │
│         └─────────────────────┴─────────────────────┘  │
│                               │                         │
│                               ▼                         │
│                        ┌──────────┐                     │
│                        │ 结果输出  │                     │
│                        └──────────┘                     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  支持模块: 知识库 | 安全检查 | 记忆系统                  │
└─────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境

复制 `.env.example` 为 `.env` 并填入配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# OpenAI配置
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4
TEMPERATURE=0.0

# 数据库配置
DATABASE_URI=mysql+pymysql://user:password@localhost:3306/database

# Agent配置
MAX_ITERATIONS=5
VERBOSE=true
```

### 3. 运行程序

```bash
python main.py
```

## 使用示例

### 简单查询

```
🤔 您的问题: 查询所有用户

💡 正在思考...

📊 查询结果:
查询返回了 100 行数据，包含以下列: id, name, email, created_at

前 10 行数据:
   id      name              email                    created_at
    1  张三      zhangsan@example.com    2024-01-01 10:00:00
    2  李四      lisi@example.com        2024-01-02 11:30:00
    ...
```

### 复杂查询

```
🤔 您的问题: 找出销售额最高的产品类别，并列出该类别下销量前5的产品

💡 正在思考...

[执行任务1]: 计算各产品类别的总销售额
[执行任务2]: 找出销售额最高的类别
[执行任务3]: 查询该类别下的产品并按销量排序

📊 查询结果:
销售额最高的产品类别是"电子产品"，总销售额为￥1,234,567。

该类别下销量前5的产品:
1. iPhone 15 Pro - 销量: 2,345台
2. MacBook Air - 销量: 1,892台
...
```

### 澄清交互

```
🤔 您的问题: 查询最近的订单

💡 正在思考...

📊 查询结果:
您提到的"最近"是指：
1. 最近7天
2. 最近30天
3. 最近一年
请选择或输入具体的时间范围。

🤔 您的问题: 最近30天

💡 正在思考...

📊 查询结果:
最近30天内共有 1,234 个订单...
```

## 命令说明

| 命令 | 说明 |
|------|------|
| `help` | 显示帮助信息 |
| `exit` / `quit` | 退出程序 |
| `clear` | 清空对话记忆 |
| `memory` | 查看记忆摘要 |
| `tables` | 查看数据库表列表 |
| `schema <表名>` | 查看指定表的详细信息 |

## 高级配置

### 添加表描述

```python
from agent import SQLAgent

agent = SQLAgent(...)

# 添加表描述
agent.knowledge_base.add_table_description(
    "users",
    "用户信息表，存储系统所有用户的基本信息"
)

# 添加列描述
agent.knowledge_base.add_column_description(
    "users",
    "created_at",
    "用户注册时间"
)
```

### 添加业务规则

```python
# 添加业务规则
agent.knowledge_base.add_business_rule(
    "活跃用户定义",
    "最近30天内有登录记录的用户视为活跃用户"
)
```

### 添加查询模式

```python
# 添加常见查询模式
agent.knowledge_base.add_query_pattern(
    "统计查询",
    "统计某个时间段内的数据",
    "SELECT COUNT(*) FROM orders WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31'"
)
```

### 调整安全级别

```python
from security import SecurityLevel

agent = SQLAgent(
    ...,
    security_level=SecurityLevel.HIGH  # LOW/MEDIUM/HIGH
)
```

## 项目结构

```
SQLAGENT2/
├── agent.py              # Agent核心逻辑
├── intent_router.py      # 意图识别与路由
├── sql_generator.py      # SQL生成器
├── query_executor.py     # 查询执行器
├── planner.py           # 任务规划器
├── clarifier.py         # 澄清器
├── knowledge_base.py    # 知识库
├── security.py          # 安全模块
├── memory.py            # 记忆系统
├── config.py            # 配置管理
├── main.py              # 主程序入口
├── requirements.txt     # 依赖包
├── .env.example         # 环境变量示例
└── README.md           # 说明文档
```

## 技术栈

- **LangChain**: Agent框架
- **OpenAI GPT-4**: 语言模型
- **SQLAlchemy**: 数据库连接和操作
- **Pandas**: 数据处理和格式化
- **Pydantic**: 数据验证和配置管理

## 注意事项

1. **安全性**: 默认只允许执行SELECT查询，可通过安全级别调整
2. **性能**: 自动添加LIMIT限制，避免返回过多数据
3. **成本**: GPT-4调用有成本，建议合理使用
4. **数据库**: 确保数据库用户有足够的查询权限

## 扩展开发

### 添加新的任务类型

在 `planner.py` 中扩展任务类型，在 `agent.py` 中添加对应的处理逻辑。

### 自定义输出格式

在 `query_executor.py` 中的 `format_results` 方法中添加新的格式类型。

### 集成其他数据源

继承 `KnowledgeBase` 类，实现新的数据源连接器。

## 常见问题

**Q: 为什么查询结果不准确？**

A: 可以通过以下方式改善：
- 添加详细的表和列描述
- 添加业务规则说明
- 提供示例查询

**Q: 如何处理大型数据库？**

A: 建议：
- 只开放必要的表给Agent
- 使用表前缀限制访问范围
- 合理设置LIMIT限制

**Q: 支持哪些数据库？**

A: 支持所有SQLAlchemy兼容的数据库，包括：
- MySQL
- PostgreSQL
- SQLite
- SQL Server
- Oracle

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

