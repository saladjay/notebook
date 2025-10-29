"""
数据库工具函数
处理中文路径、URI转换等问题
"""
import os
import sqlite3
from urllib.parse import urlparse, unquote
from langchain_community.utilities import SQLDatabase
import logging
import json
from datetime import datetime
def get_db_path_from_uri(uri: str) -> str:
    """
    从 SQLAlchemy URI 格式中提取文件路径
    支持中文路径和各种URI格式
    
    Args:
        uri: 数据库URI
        
    Returns:
        文件系统路径
        
    Examples:
        >>> get_db_path_from_uri("sqlite:///company.db")
        'company.db'
        
        >>> get_db_path_from_uri("sqlite:///D:/path/to/db.db")
        'D:\\path\\to\\db.db'
        
        >>> get_db_path_from_uri("sqlite://D:/培训/DLTools软件作业/ImageBase/db.tsdb")
        'D:\\培训\\DLTools软件作业\\ImageBase\\db.tsdb'
    """
    # 处理相对路径
    if not uri.startswith('sqlite:'):
        # 直接是文件路径
        return os.path.normpath(uri)
    
    # 移除 sqlite:// 或 sqlite:/// 前缀
    if uri.startswith('sqlite:///'):
        path = uri[10:]  # 移除 "sqlite:///"
    elif uri.startswith('sqlite://'):
        path = uri[9:]   # 移除 "sqlite://"
    else:
        path = uri
    
    # URL解码（处理 %20 等编码字符）
    path = unquote(path)
    
    # 将路径分隔符统一为系统格式
    path = path.replace('/', os.sep).replace('\\', os.sep)
    
    # 处理Windows绝对路径
    # 例如: /D:/path -> D:/path
    if len(path) > 2 and path[0] == os.sep and path[2] == ':':
        path = path[1:]
    
    return path


def connect_sqlite_safely(uri_or_path: str) -> sqlite3.Connection:
    """
    安全地连接SQLite数据库，支持中文路径
    
    Args:
        uri_or_path: 数据库URI或文件路径
        
    Returns:
        sqlite3连接对象
        
    Raises:
        FileNotFoundError: 数据库文件不存在
        sqlite3.Error: 数据库连接失败
    """
    # 获取实际文件路径
    db_path = get_db_path_from_uri(uri_or_path)
    
    # 检查文件是否存在
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    # 连接数据库
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"连接数据库失败: {str(e)}")


def fix_sqlite_uri(uri: str) -> str:
    """
    修正SQLite URI格式（处理常见错误）
    
    常见错误：
    - sqlite://D:/path  → sqlite:///D:/path  (缺少一个斜杠)
    - sqlite:///D:\\path → sqlite:///D:/path  (反斜杠)
    - D:/path/db.db     → sqlite:///D:/path/db.db (纯路径)
    
    Args:
        uri: 原始URI或路径
        
    Returns:
        修正后的正确URI格式
        
    Examples:
        >>> fix_sqlite_uri("sqlite://D:/test.db")
        'sqlite:///D:/test.db'
        
        >>> fix_sqlite_uri("D:/test.db")
        'sqlite:///D:/test.db'
    """
    if not uri or uri.strip() == '':
        raise ValueError("URI不能为空")
    
    # 如果不是sqlite URI，假设是文件路径
    if not uri.startswith('sqlite:'):
        abs_path = os.path.abspath(uri).replace('\\', '/')
        return f"sqlite:///{abs_path}"
    
    # 修正: sqlite://path -> sqlite:///path
    if uri.startswith('sqlite://') and not uri.startswith('sqlite:///'):
        path = uri[9:]  # 移除 "sqlite://"
        path = path.replace('\\', '/')
        return f"sqlite:///{path}"
    
    # 统一路径分隔符
    uri = uri.replace('\\', '/')
    
    return uri


def validate_sqlite_uri(uri: str) -> dict:
    """
    验证SQLite URI格式
    
    Args:
        uri: 数据库URI
        
    Returns:
        验证结果字典，包含：
        - is_valid: 是否有效
        - fixed_uri: 修正后的URI
        - error: 错误信息（如果有）
        - warnings: 警告信息列表
    """
    result = {
        'is_valid': False,
        'original_uri': uri,
        'fixed_uri': None,
        'error': None,
        'warnings': []
    }
    
    try:
        # 尝试修正URI
        fixed_uri = fix_sqlite_uri(uri)
        result['fixed_uri'] = fixed_uri
        
        # 检查是否有修改
        if uri != fixed_uri:
            result['warnings'].append(f"URI已自动修正: {uri} → {fixed_uri}")
        
        # 提取文件路径
        db_path = get_db_path_from_uri(fixed_uri)
        
        # 检查文件是否存在
        if not os.path.exists(db_path):
            result['warnings'].append(f"数据库文件不存在: {db_path}")
        
        result['is_valid'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_langchain_db(uri: str, auto_fix: bool = True) -> SQLDatabase:
    """
    创建LangChain的SQLDatabase对象，支持中文路径
    
    Args:
        uri: 数据库URI
        auto_fix: 是否自动修正URI格式错误
        
    Returns:
        SQLDatabase对象
        
    Raises:
        ValueError: URI格式错误且无法修正
        FileNotFoundError: 数据库文件不存在
    """
    # 自动修正URI格式
    if auto_fix:
        uri = fix_sqlite_uri(uri)
    
    try:
        db = SQLDatabase.from_uri(uri)
        return db
    except Exception as e:
        # 如果失败，尝试提取路径并重新构建URI
        try:
            db_path = get_db_path_from_uri(uri)
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"数据库文件不存在: {db_path}")
            
            # 使用绝对路径重新构建URI
            abs_path = os.path.abspath(db_path).replace('\\', '/')
            new_uri = f"sqlite:///{abs_path}"
            return SQLDatabase.from_uri(new_uri)
        except:
            raise ValueError(f"无法连接数据库，URI格式可能有误: {uri}\n原始错误: {str(e)}")


def test_connection(uri_or_path: str) -> dict:
    """
    测试数据库连接
    
    Args:
        uri_or_path: 数据库URI或路径
        
    Returns:
        包含测试结果的字典
    """
    result = {
        'success': False,
        'db_path': None,
        'exists': False,
        'can_connect': False,
        'tables': [],
        'error': None
    }
    
    try:
        # 获取路径
        db_path = get_db_path_from_uri(uri_or_path)
        result['db_path'] = db_path
        
        # 检查文件是否存在
        result['exists'] = os.path.exists(db_path)
        
        if not result['exists']:
            result['error'] = f"文件不存在: {db_path}"
            return result
        
        # 尝试连接
        conn = sqlite3.connect(db_path)
        result['can_connect'] = True
        
        # 获取表列表
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        result['tables'] = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def print_connection_test(uri_or_path: str):
    """
    打印数据库连接测试结果
    """
    print("=" * 60)
    print("🔍 数据库连接测试")
    print("=" * 60)
    
    result = test_connection(uri_or_path)
    
    print(f"\n原始URI: {uri_or_path}")
    print(f"文件路径: {result['db_path']}")
    print(f"文件存在: {'✅ 是' if result['exists'] else '❌ 否'}")
    print(f"可以连接: {'✅ 是' if result['can_connect'] else '❌ 否'}")
    
    if result['tables']:
        print(f"\n📋 数据库表 ({len(result['tables'])}个):")
        for table in result['tables']:
            print(f"  - {table}")
    
    if result['error']:
        print(f"\n❌ 错误: {result['error']}")
    
    if result['success']:
        print(f"\n✅ 连接测试成功！")
    else:
        print(f"\n❌ 连接测试失败！")
    
    print("=" * 60)
    
    return result


# ============================================
# 使用自定义数据库包装器
# ============================================

class LoggingSQLDatabase(SQLDatabase):
    """扩展SQLDatabase，添加SQL日志功能"""
    
    def __init__(self, engine, save_path: str = "sql.log", log_level: str = "INFO", log_name: str = "sql.log", *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
        self.query_log = []
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(save_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def run(self, command, fetch="all", **kwargs):
        """重写run方法，添加日志"""
        try:
            result = super().run(command, fetch=fetch, **kwargs)
            # 记录日志
            self.logger.info(json.dumps({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sql": command,
                "result": result,
                "status": "success"
            }))
            return result
        except Exception as e:
            # 记录错误
            
            self.logger.error(json.dumps({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sql": command,
                "result": str(e),
                "status": "error"
            }))
            raise
    

if __name__ == '__main__':
    """测试工具函数"""
    import sys
    
    # 从配置文件获取URI
    from config import get_config
    config = get_config()
    uri = config['database_uri']
    
    print("数据库工具函数测试\n")
    
    # 测试1: URI转路径
    print("测试1: URI转路径")
    print(f"输入URI: {uri}")
    path = get_db_path_from_uri(uri)
    print(f"输出路径: {path}\n")
    
    # 测试2: 连接测试
    print("测试2: 连接测试")
    result = print_connection_test(uri)
    
    # 测试3: LangChain连接
    if result['success']:
        print("\n测试3: LangChain SQLDatabase")
        try:
            db = get_langchain_db(uri)
            print(f"✅ LangChain连接成功")
            print(f"📋 可用表: {db.get_usable_table_names()}")
        except Exception as e:
            print(f"❌ LangChain连接失败: {str(e)}")

