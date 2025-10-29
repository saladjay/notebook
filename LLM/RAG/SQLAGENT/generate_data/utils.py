import sqlite3
from config import get_config

config = get_config()

def _fix_sqlite_uri(uri: str) -> str:
    """
    修正SQLite URI格式（自动处理常见错误）
    
    常见错误：
    - sqlite://D:/path  (错误：只有2个斜杠)
    - sqlite:///D:\\path (错误：使用反斜杠)
    
    正确格式：
    - sqlite:///company.db        (相对路径)
    - sqlite:///D:/path/to/db.db  (绝对路径，3个斜杠)
    
    Args:
        uri: 原始URI或文件路径
        
    Returns:
        修正后的URI
    """
    if not uri or uri.strip() == '':
        raise ValueError("数据库URI不能为空")
    
    # 如果不是sqlite URI，假设是文件路径
    if not uri.startswith('sqlite:'):
        abs_path = os.path.abspath(uri).replace('\\', '/')
        return f"sqlite:///{abs_path}"
    
    # 修正常见的格式错误：sqlite://path (2个斜杠) -> sqlite:///path (3个斜杠)
    if uri.startswith('sqlite://') and not uri.startswith('sqlite:///'):
        path = uri[9:]  # 移除 "sqlite://"
        # 统一路径分隔符为 /
        path = path.replace('\\', '/')
        return f"sqlite:///{path}"
    
    # 统一路径分隔符（Windows的\改为/）
    uri = uri.replace('\\', '/')
    
    return uri

def get_db_path_from_uri(db_uri: str) -> str:
    db_path = _fix_sqlite_uri(db_uri)
    return db_path.replace("sqlite:///", "")

