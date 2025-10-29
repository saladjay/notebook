# 快速启动指南

本指南将帮助您在5分钟内启动SQL Agent。

## 步骤1: 安装依赖

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

## 步骤2: 配置环境变量

复制 `env_template.txt` 的内容到新文件 `.env`：

```bash
# Windows PowerShell:
Copy-Item env_template.txt .env

# Linux/Mac:
cp env_template.txt .env
```

编辑 `.env` 文件，填入你的配置：

```ini
# OpenAI配置
OPENAI_API_KEY=sk-your-actual-api-key-here
MODEL_NAME=gpt-4

# 数据库配置（选择一个）
# MySQL:
DATABASE_URI=mysql+pymysql://username:password@localhost:3306/database_name

# 或 PostgreSQL:
# DATABASE_URI=postgresql://username:password@localhost:5432/database_name

# 或 SQLite (用于测试):
# DATABASE_URI=sqlite:///test.db
```

## 步骤3: 测试数据库连接

使用Python测试数据库连接：

```python
from sqlalchemy import create_engine

# 使用你的DATABASE_URI
engine = create_engine("your_database_uri_here")

try:
    with engine.connect() as conn:
        print("✓ 数据库连接成功!")
except Exception as e:
    print(f"✗ 数据库连接失败: {e}")
```

## 步骤4: 运行测试（可选）

```bash
python test_agent.py
```

如果所有测试通过，说明环境配置正确。

## 步骤5: 启动Agent

```bash
python main.py
```

你应该看到欢迎界面：

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║              SQL Agent - 智能数据查询助手                   ║
║                                                            ║
║  支持自然语言查询、复杂多步推理、智能澄清等功能              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

## 步骤6: 开始使用

### 查看可用的表

```
🤔 您的问题: tables
```

### 查看表结构

```
🤔 您的问题: schema users
```

### 进行简单查询

```
🤔 您的问题: 查询所有用户
```

### 进行复杂查询

```
🤔 您的问题: 统计每个月的订单数量，并找出订单最多的月份
```

## 常见问题

### Q1: 提示"ModuleNotFoundError"

**解决方案**: 确保已激活虚拟环境并安装了所有依赖。

```bash
# 重新安装依赖
pip install -r requirements.txt
```

### Q2: 数据库连接失败

**解决方案**: 检查以下项目：

1. 数据库服务是否运行
2. 用户名和密码是否正确
3. 数据库名称是否存在
4. 网络连接是否正常

### Q3: OpenAI API错误

**解决方案**: 

1. 检查API密钥是否有效
2. 检查账户是否有余额
3. 如果在中国，可能需要配置代理或使用国内镜像

### Q4: 查询结果不准确

**解决方案**: 

1. 添加表和列的描述信息
2. 添加业务规则
3. 提供查询示例
4. 使用更明确的问题描述

参考 `example_usage.py` 中的示例。

## 下一步

- 查看 [README.md](README.md) 了解完整功能
- 查看 [example_usage.py](example_usage.py) 学习高级用法
- 根据需求配置知识库和安全规则

## 获取帮助

遇到问题？

1. 查看 `README.md` 的常见问题部分
2. 运行 `python main.py` 后输入 `help` 查看命令
3. 查看代码注释和文档字符串

祝使用愉快! 🎉

