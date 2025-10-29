"""
SQL Agent with SQL Logging - 记录所有执行的SQL语句和结果
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from config import get_config
import re
from datetime import datetime


# ============================================
# 方法1: 使用回调系统捕获SQL
# ============================================

class SQLLoggingCallback(BaseCallbackHandler):
    """自定义回调处理器，用于记录SQL语句和结果"""
    
    def __init__(self):
        self.sql_logs = []
        self.current_sql = None
        
    def on_tool_start(self, serialized, input_str, **kwargs):
        """工具开始执行时调用"""
        tool_name = serialized.get("name", "")
        if "sql" in tool_name.lower():
            # 提取SQL语句
            sql = self._extract_sql(input_str)
            if sql:
                self.current_sql = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sql": sql,
                    "tool": tool_name,
                    "result": None
                }
                print(f"\n{'='*80}")
                print(f"🔍 执行SQL语句:")
                print(f"{'='*80}")
                print(f"{sql}")
                print(f"{'='*80}\n")
    
    def on_tool_end(self, output, **kwargs):
        """工具执行结束时调用"""
        if self.current_sql:
            self.current_sql["result"] = output
            self.sql_logs.append(self.current_sql)
            
            print(f"\n{'='*80}")
            print(f"✅ 执行结果:")
            print(f"{'='*80}")
            print(f"{output}")
            print(f"{'='*80}\n")
            
            self.current_sql = None
    
    def on_tool_error(self, error, **kwargs):
        """工具执行错误时调用"""
        if self.current_sql:
            self.current_sql["result"] = f"ERROR: {str(error)}"
            self.sql_logs.append(self.current_sql)
            
            print(f"\n{'='*80}")
            print(f"❌ 执行错误:")
            print(f"{'='*80}")
            print(f"{str(error)}")
            print(f"{'='*80}\n")
            
            self.current_sql = None
    
    def _extract_sql(self, input_str):
        """从输入字符串中提取SQL语句"""
        # 尝试多种模式提取SQL
        patterns = [
            r'SELECT.*?(?:;|$)',
            r'INSERT.*?(?:;|$)',
            r'UPDATE.*?(?:;|$)',
            r'DELETE.*?(?:;|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(input_str), re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0).strip()
        
        return str(input_str).strip()
    
    def get_sql_history(self):
        """获取所有SQL执行历史"""
        return self.sql_logs
    
    def print_summary(self):
        """打印SQL执行摘要"""
        print(f"\n{'='*80}")
        print(f"📊 SQL执行摘要")
        print(f"{'='*80}")
        print(f"总共执行了 {len(self.sql_logs)} 条SQL语句\n")
        
        for i, log in enumerate(self.sql_logs, 1):
            print(f"[{i}] {log['timestamp']}")
            print(f"    SQL: {log['sql'][:100]}...")
            if log['result']:
                result_preview = str(log['result'])[:100]
                print(f"    结果: {result_preview}...")
            print()


# ============================================
# 方法2: 使用自定义数据库包装器
# ============================================

class LoggingSQLDatabase(SQLDatabase):
    """扩展SQLDatabase，添加SQL日志功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_log = []
    
    def run(self, command, fetch="all", **kwargs):
        """重写run方法，添加日志"""
        print(f"\n{'='*80}")
        print(f"🔍 执行SQL:")
        print(f"{'='*80}")
        print(f"{command}")
        print(f"{'='*80}")
        
        try:
            result = super().run(command, fetch=fetch, **kwargs)
            
            print(f"\n{'='*80}")
            print(f"✅ 结果:")
            print(f"{'='*80}")
            print(f"{result}")
            print(f"{'='*80}\n")
            
            # 记录日志
            self.query_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sql": command,
                "result": result,
                "status": "success"
            })
            
            return result
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ 错误:")
            print(f"{'='*80}")
            print(f"{str(e)}")
            print(f"{'='*80}\n")
            
            # 记录错误
            self.query_log.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sql": command,
                "result": str(e),
                "status": "error"
            })
            
            raise
    
    def get_query_log(self):
        """获取查询日志"""
        return self.query_log
    
    def print_query_summary(self):
        """打印查询摘要"""
        print(f"\n{'='*80}")
        print(f"📊 SQL查询摘要")
        print(f"{'='*80}")
        
        success_count = sum(1 for log in self.query_log if log['status'] == 'success')
        error_count = sum(1 for log in self.query_log if log['status'] == 'error')
        
        print(f"总查询数: {len(self.query_log)}")
        print(f"成功: {success_count} ✅")
        print(f"失败: {error_count} ❌")
        print(f"{'='*80}\n")
        
        for i, log in enumerate(self.query_log, 1):
            status_icon = "✅" if log['status'] == 'success' else "❌"
            print(f"{status_icon} [{i}] {log['timestamp']}")
            print(f"    SQL: {log['sql']}")
            print(f"    结果: {str(log['result'])[:200]}")
            print()


# ============================================
# 创建带日志的Agent
# ============================================

def create_agent_with_callback():
    """方法1: 使用回调系统"""
    config = get_config()
    
    # 初始化LLM
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    # 初始化数据库
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    # 创建回调处理器
    sql_callback = SQLLoggingCallback()
    
    # 创建Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[sql_callback]  # 添加回调
    )
    
    return agent, sql_callback


def create_agent_with_logging_db():
    """方法2: 使用自定义数据库包装器"""
    config = get_config()
    
    # 初始化LLM
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    # 使用自定义的数据库类
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = LoggingSQLDatabase.from_uri(db_uri)
    
    # 创建Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent, db


# ============================================
# 方法3: 保存到文件
# ============================================

def save_sql_log_to_file(sql_logs, filename="sql_execution_log.txt"):
    """将SQL日志保存到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SQL执行日志\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for i, log in enumerate(sql_logs, 1):
            f.write(f"[{i}] {log.get('timestamp', 'N/A')}\n")
            f.write(f"SQL语句:\n{log.get('sql', 'N/A')}\n\n")
            f.write(f"执行结果:\n{log.get('result', 'N/A')}\n")
            f.write("-"*80 + "\n\n")
    
    print(f"✅ SQL日志已保存到: {filename}")


# ============================================
# 测试函数
# ============================================

def test_method1():
    """测试方法1: 回调系统"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🔍 方法1: 使用回调系统记录SQL                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建带回调的Agent
    agent, sql_callback = create_agent_with_callback()
    
    # 测试查询
    test_questions = [
        "数据库里有几个表？",
        "Image表有多少行数据？",
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"❓ 问题: {question}")
        print(f"{'='*80}\n")
        
        try:
            result = agent.invoke({"input": question})
            print(f"\n💡 最终答案: {result.get('output', result)}\n")
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}\n")
    
    # 打印摘要
    sql_callback.print_summary()
    
    # 保存到文件
    save_sql_log_to_file(sql_callback.get_sql_history())


def test_method2():
    """测试方法2: 自定义数据库包装器"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🔍 方法2: 使用自定义数据库包装器记录SQL                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建带日志的Agent
    agent, db = create_agent_with_logging_db()
    
    # 测试查询
    test_questions = [
        "数据库里有几个表？",
        "Image表有多少行数据？",
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"❓ 问题: {question}")
        print(f"{'='*80}\n")
        
        try:
            result = agent.invoke({"input": question})
            print(f"\n💡 最终答案: {result.get('output', result)}\n")
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}\n")
    
    # 打印摘要
    db.print_query_summary()
    
    # 保存到文件
    save_sql_log_to_file(db.get_query_log(), "sql_execution_log_method2.txt")


def main():
    """主函数"""
    print("\n选择测试方法:")
    print("  [1] 方法1: 使用回调系统（推荐）")
    print("  [2] 方法2: 使用自定义数据库包装器")
    print("  [0] 退出")
    
    choice = input("\n请选择 (0-2): ").strip()
    
    if choice == '0':
        print("👋 再见！")
        return
    elif choice == '1':
        test_method1()
    elif choice == '2':
        test_method2()
    else:
        print("❌ 无效选项")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

