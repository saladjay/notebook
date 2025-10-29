import sqlite3
import time
from config import get_config
from db_utils import connect_sqlite_safely, print_connection_test
import functools

from sql_agent_demo import SQLAgentDemo

config = get_config()

# 获取数据库URI
db_uri = config['database_uri']

# 测试并连接数据库（自动处理中文路径）
print("=" * 60)
print("🔍 数据库基准测试")
print("=" * 60)

# 连接数据库（支持中文路径）
try:
    sqlite_conn = connect_sqlite_safely(db_uri)
    print(f"✅ 数据库连接成功！")
except FileNotFoundError as e:
    print(f"❌ 错误: {str(e)}")
    print("\n💡 提示:")
    print("  1. 检查 config.py 中的 DATABASE_URI 配置")
    print("  2. 如果是新项目，运行: python init_database.py")
    print("  3. 如果路径包含中文，确保使用正确的格式")
    print("\n运行 'python db_utils.py' 查看详细诊断")
    exit(1)
except Exception as e:
    print(f"❌ 连接失败: {str(e)}")
    exit(1)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"函数 {func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result
    return wrapper

def time_query(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"query 执行耗时: {end - start:.4f} 秒")
        return result
    return wrapper

@time_query
def run_query(agent, question):
    return agent.query(question)

@measure_time
def get_tables(conn):
    """获取所有表名"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return cursor.fetchall()

@measure_time
def get_table_data(conn, table_name):
    """获取表数据"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    return cursor.fetchall()

if __name__ == "__main__":
    print("\n测试查询性能...")
    print("-" * 60)
    
    # 获取所有表
    tables = get_tables(sqlite_conn)
    print(f"数据库表: {[t[0] for t in tables]}")
    
    sql_agent = SQLAgentDemo()

    result = run_query(sql_agent, "请帮我查询所有表名/nothink")
    print(result)
    
    print("-" * 60)
    print("✅ 基准测试完成！")
    
    sqlite_conn.close()

