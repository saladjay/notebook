"""
SQL Agent主程序
提供命令行交互界面
"""
import sys
from typing import Optional
from config import settings
from agent import SQLAgent
from security import SecurityLevel


def print_banner():
    """打印欢迎信息"""
    banner = """
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║              SQL Agent - 智能数据查询助手                   ║
║                                                            ║
║  支持自然语言查询、复杂多步推理、智能澄清等功能              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("输入 'help' 查看帮助信息")
    print("输入 'exit' 或 'quit' 退出程序")
    print("=" * 60)


def print_help():
    """打印帮助信息"""
    help_text = """
可用命令：
  help          - 显示此帮助信息
  exit/quit     - 退出程序
  clear         - 清空对话记忆
  memory        - 查看记忆摘要
  tables        - 查看数据库表列表
  schema <表名> - 查看指定表的详细信息
  
直接输入问题即可开始查询，例如：
  - 查询所有用户
  - 统计过去一个月的订单数量
  - 找出销售额最高的产品类别
"""
    print(help_text)


def initialize_agent() -> Optional[SQLAgent]:
    """初始化Agent"""
    try:
        print("正在初始化SQL Agent...")
        
        # 根据配置创建Agent
        agent = SQLAgent(
            database_uri=settings.database_uri,
            openai_api_key=settings.openai_api_key,
            model_name=settings.model_name,
            temperature=settings.temperature,
            security_level=SecurityLevel.MEDIUM,
            verbose=settings.verbose
        )
        
        print("初始化完成!\n")
        return agent
        
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        print("\n请检查:")
        print("1. .env文件是否存在并配置正确")
        print("2. 数据库连接是否正常")
        print("3. OpenAI API密钥是否有效")
        return None


def handle_command(agent: SQLAgent, command: str) -> bool:
    """
    处理特殊命令
    
    Returns:
        True表示继续运行，False表示退出
    """
    command = command.strip().lower()
    
    if command in ['exit', 'quit']:
        print("\n感谢使用，再见！")
        return False
    
    elif command == 'help':
        print_help()
    
    elif command == 'clear':
        agent.clear_memory()
        print("记忆已清空")
    
    elif command == 'memory':
        summary = agent.get_memory_summary()
        print("\n记忆摘要:")
        print(f"  对话轮数: {summary['total_conversations']}")
        print(f"  执行次数: {summary['total_executions']}")
        print(f"  SQL查询: {summary['total_sql_queries']} "
              f"(成功: {summary['successful_sql_queries']})")
        print(f"  知识条目: {summary['knowledge_items']}")
    
    elif command == 'tables':
        tables = agent.knowledge_base.get_table_names()
        print("\n数据库表列表:")
        for table in tables:
            desc = agent.knowledge_base.table_descriptions.get(table, "")
            print(f"  - {table}", end="")
            if desc:
                print(f": {desc}", end="")
            print()
    
    elif command.startswith('schema '):
        table_name = command[7:].strip()
        try:
            info = agent.knowledge_base.get_table_info(table_name)
            print(f"\n表 {table_name} 的信息:")
            print(f"描述: {info.get('description', '无')}")
            print("\n列信息:")
            for col in info['columns']:
                print(f"  - {col['name']} ({col['type']})")
        except Exception as e:
            print(f"获取表信息失败: {str(e)}")
    
    else:
        # 不是特殊命令，作为查询处理
        return True
    
    return True


def main():
    """主函数"""
    print_banner()
    
    # 初始化Agent
    agent = initialize_agent()
    if not agent:
        sys.exit(1)
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n🤔 您的问题: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ['help', 'exit', 'quit', 'clear', 'memory', 'tables'] or \
               user_input.lower().startswith('schema '):
                should_continue = handle_command(agent, user_input)
                if not should_continue:
                    break
                continue
            
            # 处理查询
            print("\n💡 正在思考...\n")
            result = agent.query(user_input)
            print("\n" + "="*60)
            print("📊 查询结果:")
            print("="*60)
            print(result)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            print("请重试或输入 'help' 查看帮助")


if __name__ == "__main__":
    main()


