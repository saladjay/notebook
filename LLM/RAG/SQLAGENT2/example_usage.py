"""
使用示例
演示如何使用SQL Agent的各种功能
"""
from agent import SQLAgent
from security import SecurityLevel


def example_basic_usage():
    """基本使用示例"""
    print("="*60)
    print("示例1: 基本使用")
    print("="*60)
    
    # 创建Agent
    agent = SQLAgent(
        database_uri="mysql+pymysql://user:password@localhost:3306/mydb",
        openai_api_key="your-api-key",
        model_name="gpt-4",
        security_level=SecurityLevel.MEDIUM,
        verbose=True
    )
    
    # 简单查询
    result = agent.query("查询所有用户")
    print(result)


def example_with_knowledge_base():
    """使用知识库示例"""
    print("="*60)
    print("示例2: 配置知识库")
    print("="*60)
    
    agent = SQLAgent(
        database_uri="mysql+pymysql://user:password@localhost:3306/mydb",
        openai_api_key="your-api-key"
    )
    
    # 添加表描述
    agent.knowledge_base.add_table_description(
        "users",
        "用户信息表，存储系统所有用户的基本信息"
    )
    
    agent.knowledge_base.add_table_description(
        "orders",
        "订单表，记录所有订单信息"
    )
    
    # 添加列描述
    agent.knowledge_base.add_column_description(
        "users",
        "created_at",
        "用户注册时间"
    )
    
    agent.knowledge_base.add_column_description(
        "users",
        "status",
        "用户状态：active(活跃), inactive(未激活), banned(禁用)"
    )
    
    # 添加业务规则
    agent.knowledge_base.add_business_rule(
        "活跃用户",
        "最近30天内有登录记录的用户视为活跃用户"
    )
    
    agent.knowledge_base.add_business_rule(
        "订单状态",
        "订单状态包括：pending(待处理), processing(处理中), completed(已完成), cancelled(已取消)"
    )
    
    # 添加查询模式
    agent.knowledge_base.add_query_pattern(
        "时间范围统计",
        "统计某个时间段内的数据量",
        "SELECT COUNT(*) FROM orders WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31'"
    )
    
    # 现在查询会使用这些知识
    result = agent.query("统计活跃用户数量")
    print(result)


def example_complex_query():
    """复杂查询示例"""
    print("="*60)
    print("示例3: 复杂多步查询")
    print("="*60)
    
    agent = SQLAgent(
        database_uri="mysql+pymysql://user:password@localhost:3306/mydb",
        openai_api_key="your-api-key"
    )
    
    # 复杂查询会自动分解为多个子任务
    result = agent.query(
        "找出销售额最高的产品类别，并列出该类别下销量前5的产品，"
        "同时显示每个产品的平均评分"
    )
    print(result)


def example_security_settings():
    """安全设置示例"""
    print("="*60)
    print("示例4: 安全设置")
    print("="*60)
    
    # 高安全级别：只允许SELECT
    agent_high = SQLAgent(
        database_uri="mysql+pymysql://user:password@localhost:3306/mydb",
        openai_api_key="your-api-key",
        security_level=SecurityLevel.HIGH
    )
    
    # 设置允许访问的表前缀
    agent_high.security.set_allowed_tables(["user_", "order_"])
    
    # 设置最大返回行数
    agent_high.security.set_max_rows(100)
    
    result = agent_high.query("查询用户信息")
    print(result)


def example_memory_usage():
    """记忆系统使用示例"""
    print("="*60)
    print("示例5: 记忆系统")
    print("="*60)
    
    agent = SQLAgent(
        database_uri="mysql+pymysql://user:password@localhost:3306/mydb",
        openai_api_key="your-api-key"
    )
    
    # 执行几次查询
    agent.query("查询所有用户")
    agent.query("统计订单数量")
    agent.query("查询最新的10个订单")
    
    # 查看记忆摘要
    summary = agent.get_memory_summary()
    print("\n记忆摘要:")
    print(f"  对话轮数: {summary['total_conversations']}")
    print(f"  SQL查询: {summary['total_sql_queries']}")
    print(f"  成功查询: {summary['successful_sql_queries']}")
    
    # 获取最近的对话
    recent_conversations = agent.memory.get_recent_conversations(n=3)
    print("\n最近的对话:")
    for i, conv in enumerate(recent_conversations, 1):
        print(f"\n对话{i}:")
        print(f"  用户: {conv['user']}")
        print(f"  助手: {conv['assistant'][:100]}...")
    
    # 清空记忆
    agent.clear_memory()
    print("\n记忆已清空")


def example_interactive_clarification():
    """交互式澄清示例"""
    print("="*60)
    print("示例6: 交互式澄清")
    print("="*60)
    
    agent = SQLAgent(
        database_uri="mysql+pymysql://user:password@localhost:3306/mydb",
        openai_api_key="your-api-key"
    )
    
    # 提出一个模糊的问题
    result1 = agent.query("查询最近的订单")
    print(f"\nAgent回复: {result1}")
    
    # Agent会要求澄清"最近"的含义，用户回答
    result2 = agent.query("最近7天")
    print(f"\nAgent回复: {result2}")


def example_with_postgresql():
    """使用PostgreSQL的示例"""
    print("="*60)
    print("示例7: 使用PostgreSQL")
    print("="*60)
    
    agent = SQLAgent(
        database_uri="postgresql://user:password@localhost:5432/mydb",
        openai_api_key="your-api-key"
    )
    
    result = agent.query("查询所有表的记录数")
    print(result)


def example_batch_queries():
    """批量查询示例"""
    print("="*60)
    print("示例8: 批量查询")
    print("="*60)
    
    agent = SQLAgent(
        database_uri="mysql+pymysql://user:password@localhost:3306/mydb",
        openai_api_key="your-api-key",
        verbose=False  # 关闭详细输出
    )
    
    questions = [
        "统计用户总数",
        "统计订单总数",
        "查询今天的新用户数",
        "查询本月的订单金额",
        "查询最热门的产品类别"
    ]
    
    results = {}
    for question in questions:
        print(f"\n处理问题: {question}")
        result = agent.query(question)
        results[question] = result
        print(f"结果: {result[:100]}...")
    
    return results


if __name__ == "__main__":
    print("""
SQL Agent 使用示例
==================

本文件包含多个使用示例，演示SQL Agent的各种功能。

注意：运行前请先配置好.env文件和数据库连接。

示例列表：
1. 基本使用
2. 配置知识库
3. 复杂多步查询
4. 安全设置
5. 记忆系统
6. 交互式澄清
7. 使用PostgreSQL
8. 批量查询

要运行特定示例，请取消注释相应的函数调用。
""")
    
    # 取消注释以运行示例
    # example_basic_usage()
    # example_with_knowledge_base()
    # example_complex_query()
    # example_security_settings()
    # example_memory_usage()
    # example_interactive_clarification()
    # example_with_postgresql()
    # example_batch_queries()





