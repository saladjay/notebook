"""
配置文件 - 可以在这里修改各种设置
"""

# ============================================
# Ollama 配置
# ============================================

# Ollama模型名称
# 可选项: llama3.2, qwen2.5, mistral, deepseek-coder, tinyllama 等
# OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_MODEL = "qwen3:14b"
# OLLAMA_MODEL = "sqlcoder:7b"

# 温度参数 (0-1)
# 0: 更确定性的输出
# 1: 更随机的输出
TEMPERATURE = 0

# Ollama服务地址（如果使用远程Ollama服务）
# 默认: None (使用本地服务)
OLLAMA_BASE_URL = "http://192.168.2.193:11434"  # 例如: "http://localhost:11434"

# ============================================
# 数据库配置
# ============================================

# 数据库URI
# SQLite: sqlite:///database.db
# MySQL: mysql+pymysql://user:pass@host/dbname
# PostgreSQL: postgresql://user:pass@host/dbname
# DATABASE_URI = r"sqlite:///D:/培训/DLTools软件作业/ImageBase/默认图库-DDet.tsdb"
DATABASE_URI = r"sqlite:///D:/github/notebook/LLM/RAG/SQLAGENT/example_dataset.db"

# ============================================
# Agent 配置
# ============================================

# 是否显示详细执行过程
VERBOSE = True

# 是否处理解析错误
HANDLE_PARSING_ERRORS = True

# Agent类型
# 可选: ZERO_SHOT_REACT_DESCRIPTION, CONVERSATIONAL_REACT_DESCRIPTION
AGENT_TYPE = "ZERO_SHOT_REACT_DESCRIPTION"

# 最大迭代次数
MAX_ITERATIONS = 15

# ============================================
# 查询配置
# ============================================

# 默认查询超时时间（秒）
QUERY_TIMEOUT = 60

# 最大返回结果数
MAX_RESULTS = 100

# ============================================
# 日志配置
# ============================================

# 是否启用日志
ENABLE_LOGGING = True

# 日志级别 (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = "INFO"

# 日志文件路径
LOG_FILE = "sql_agent.log"

# ============================================
# UI 配置
# ============================================

# 控制台颜色（用于美化输出）
COLORS = {
    'header': '=' * 60,
    'subheader': '-' * 60,
    'success': '✅',
    'error': '❌',
    'info': 'ℹ️',
    'warning': '⚠️',
    'question': '❓',
    'answer': '💡',
    'search': '🔍',
}

# ============================================
# 预设查询示例
# ============================================

EXAMPLE_QUERIES = [
    "技术部有多少员工？",
    "工资最高的3名员工是谁？他们的工资是多少？",
    "每个部门的平均工资是多少？",
    "哪些员工的工资高于15000？",
    "列出2020年入职的所有员工",
    "年龄最大的员工在哪个部门？",
]

# ============================================
# 高级配置
# ============================================

# 是否启用缓存（提高重复查询速度）
ENABLE_CACHE = False

# 缓存过期时间（秒）
CACHE_TTL = 3600

# 是否启用SQL验证（检查生成的SQL是否安全）
ENABLE_SQL_VALIDATION = True

# 禁止的SQL关键字（安全考虑）
FORBIDDEN_SQL_KEYWORDS = [
    'DROP',
    'DELETE',
    'TRUNCATE',
    'ALTER',
    'CREATE',
    'GRANT',
    'REVOKE',
]

# ============================================
# 提示词模板（高级用户可自定义）
# ============================================

CUSTOM_PROMPT_TEMPLATE = None  # None表示使用默认模板

# 如果需要自定义，可以这样：
# CUSTOM_PROMPT_TEMPLATE = """
# You are a SQL expert. Given a question and database schema, write SQL query.
# 
# Schema: {schema}
# Question: {question}
# 
# SQL Query:
# """


def get_config():
    """获取配置字典"""
    return {
        'ollama_model': OLLAMA_MODEL,
        'temperature': TEMPERATURE,
        'ollama_base_url': OLLAMA_BASE_URL,
        'database_uri': DATABASE_URI,
        'verbose': VERBOSE,
        'handle_parsing_errors': HANDLE_PARSING_ERRORS,
        'agent_type': AGENT_TYPE,
        'max_iterations': MAX_ITERATIONS,
        'query_timeout': QUERY_TIMEOUT,
        'max_results': MAX_RESULTS,
        'enable_logging': ENABLE_LOGGING,
        'log_level': LOG_LEVEL,
        'log_file': LOG_FILE,
        'example_queries': EXAMPLE_QUERIES,
        'enable_cache': ENABLE_CACHE,
        'cache_ttl': CACHE_TTL,
        'enable_sql_validation': ENABLE_SQL_VALIDATION,
        'forbidden_sql_keywords': FORBIDDEN_SQL_KEYWORDS,
    }


def print_config():
    """打印当前配置"""
    config = get_config()
    print("=" * 60)
    print("📋 当前配置")
    print("=" * 60)
    for key, value in config.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} 项")
        else:
            print(f"{key}: {value}")
    print("=" * 60)


if __name__ == '__main__':
    print_config()

