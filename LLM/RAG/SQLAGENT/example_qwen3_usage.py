"""
Qwen3 SQL Agent使用示例
展示如何使用修复后的Agent
"""

def example_basic():
    """基础示例：最简单的用法"""
    print("=" * 80)
    print("示例1: 基础用法（推荐）")
    print("=" * 80)
    
    from sql_agent.sql_agent_qwen import SQLAgentQwen
    
    # 创建Agent（自动处理Qwen3的思考模式）
    agent = SQLAgentQwen(
        model_name="qwen3:7b",  # 根据您的模型名称修改
        database_uri="company.db"  # 根据您的数据库路径修改
    )
    
    # 执行查询
    result = agent.query("数据库中有哪些表？")
    print(f"\n结果: {result['output']}\n")


def example_multiple_queries():
    """示例：批量查询"""
    print("=" * 80)
    print("示例2: 批量查询")
    print("=" * 80)
    
    from sql_agent.sql_agent_qwen import SQLAgentQwen
    
    agent = SQLAgentQwen(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    
    questions = [
        "数据库中有哪些表？",
        "employees表有多少条记录？",
        "薪资最高的员工是谁？",
        "平均薪资是多少？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        try:
            result = agent.query(question)
            print(f"回答: {result['output']}")
        except Exception as e:
            print(f"错误: {str(e)}")
        print("-" * 60)


def example_with_callbacks():
    """示例：使用回调记录执行过程"""
    print("=" * 80)
    print("示例3: 使用回调函数")
    print("=" * 80)
    
    from sql_agent.sql_agent_qwen import SQLAgentQwen
    from callbacks import ChainInteractionLogger
    
    agent = SQLAgentQwen(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    
    # 创建回调以记录执行过程
    callback = ChainInteractionLogger(
        log_file="qwen3_execution.log",
        console_output=True
    )
    
    # 执行查询并记录
    result = agent.query(
        "查询所有部门的员工数量",
        callback=[callback]
    )
    
    print(f"\n结果: {result['output']}")
    print("\n执行日志已保存到: qwen3_execution.log")


def example_error_handling():
    """示例：错误处理"""
    print("=" * 80)
    print("示例4: 错误处理")
    print("=" * 80)
    
    from sql_agent.sql_agent_qwen import SQLAgentQwen
    
    try:
        agent = SQLAgentQwen(
            model_name="qwen3:7b",
            database_uri="company.db"
        )
        
        # 执行可能出错的查询
        result = agent.query("这是一个可能很难的问题")
        print(f"成功: {result['output']}")
        
    except Exception as e:
        print(f"捕获到错误: {str(e)}")
        print("\n可能的原因：")
        print("1. 模型无法理解问题")
        print("2. 数据库中没有相关数据")
        print("3. 查询太复杂，超过最大迭代次数")


def example_compare_methods():
    """示例：对比不同方法"""
    print("=" * 80)
    print("示例5: 对比简单方法vs高级方法")
    print("=" * 80)
    
    question = "数据库中有几个表？"
    
    # 方法1: 简单方法（自动重试）
    print("\n方法1: SQLAgentQwen（自动重试）")
    print("-" * 60)
    from sql_agent.sql_agent_qwen import SQLAgentQwen
    
    agent1 = SQLAgentQwen(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    result1 = agent1.query(question)
    print(f"结果: {result1['output']}")
    
    # 方法2: 高级方法（自定义解析器）
    print("\n方法2: SQLAgentQwenAdvanced（自定义解析器）")
    print("-" * 60)
    from sql_agent.sql_agent_qwen import SQLAgentQwenAdvanced
    
    agent2 = SQLAgentQwenAdvanced(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    result2 = agent2.query(question)
    print(f"结果: {result2['output']}")


def example_complex_query():
    """示例：复杂查询"""
    print("=" * 80)
    print("示例6: 复杂查询")
    print("=" * 80)
    
    from sql_agent.sql_agent_qwen import SQLAgentQwen
    
    agent = SQLAgentQwen(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    
    # 执行需要多步推理的复杂查询
    complex_questions = [
        "找出薪资高于所在部门平均薪资的员工",
        "统计每个部门的最高薪资和最低薪资",
        "找出薪资排名前10%的员工所在的部门分布"
    ]
    
    for question in complex_questions:
        print(f"\n问题: {question}")
        print("-" * 60)
        try:
            result = agent.query(question)
            print(f"回答: {result['output']}")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")


def main():
    """运行所有示例"""
    import sys
    
    print("\n🎯 Qwen3 SQL Agent 使用示例集\n")
    
    examples = {
        "1": ("基础用法", example_basic),
        "2": ("批量查询", example_multiple_queries),
        "3": ("使用回调", example_with_callbacks),
        "4": ("错误处理", example_error_handling),
        "5": ("对比方法", example_compare_methods),
        "6": ("复杂查询", example_complex_query),
    }
    
    if len(sys.argv) > 1:
        # 运行指定示例
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            print(f"运行示例 {choice}: {name}\n")
            func()
        else:
            print(f"未知示例: {choice}")
            print("可用示例: 1, 2, 3, 4, 5, 6")
    else:
        # 显示菜单
        print("请选择要运行的示例：")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print("\n运行方式: python example_qwen3_usage.py [示例编号]")
        print("例如: python example_qwen3_usage.py 1")
        print("\n或者直接运行示例1:")
        example_basic()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

