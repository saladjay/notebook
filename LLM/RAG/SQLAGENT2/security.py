"""
安全模块
用于SQL注入检测、权限控制、查询限制等
"""
import re
from typing import Optional, Tuple
from enum import Enum


class SecurityLevel(Enum):
    """安全级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SecurityModule:
    """安全模块"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """
        初始化安全模块
        
        Args:
            security_level: 安全级别
        """
        self.security_level = security_level
        
        # 禁止的SQL关键字
        self.forbidden_keywords = {
            SecurityLevel.HIGH: [
                "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", 
                "UPDATE", "INSERT", "EXEC", "EXECUTE", "GRANT", 
                "REVOKE", "SHUTDOWN", "RENAME"
            ],
            SecurityLevel.MEDIUM: [
                "DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT", 
                "REVOKE", "SHUTDOWN", "RENAME"
            ],
            SecurityLevel.LOW: [
                "DROP", "SHUTDOWN"
            ]
        }
        
        # 允许的表前缀（可选）
        self.allowed_table_prefixes: Optional[list] = None
        
        # 最大返回行数
        self.max_rows = 1000
    
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        验证SQL的安全性
        
        Args:
            sql: SQL语句
            
        Returns:
            (是否安全, 错误信息)
        """
        sql_upper = sql.upper()
        
        # 1. 检查禁止的关键字
        forbidden = self.forbidden_keywords[self.security_level]
        for keyword in forbidden:
            # 使用正则匹配完整单词
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, sql_upper):
                return False, f"检测到禁止的SQL操作: {keyword}"
        
        # 2. 检查SQL注入模式
        injection_patterns = [
            r";\s*(DROP|DELETE|UPDATE|INSERT)",  # 多语句注入
            r"--",  # 注释注入
            r"/\*.*\*/",  # 块注释
            r"UNION\s+SELECT",  # UNION注入
            r"OR\s+1\s*=\s*1",  # 永真条件
            r"OR\s+'1'\s*=\s*'1'",  # 永真条件
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql_upper):
                return False, f"检测到可疑的SQL注入模式"
        
        # 3. 检查是否只有SELECT语句
        if self.security_level == SecurityLevel.HIGH:
            if not sql_upper.strip().startswith("SELECT"):
                return False, "只允许执行SELECT查询"
        
        # 4. 检查表名前缀（如果设置了）
        if self.allowed_table_prefixes:
            tables = self._extract_table_names(sql)
            for table in tables:
                if not any(table.startswith(prefix) for prefix in self.allowed_table_prefixes):
                    return False, f"不允许访问表: {table}"
        
        return True, None
    
    def _extract_table_names(self, sql: str) -> list:
        """从SQL中提取表名（简单实现）"""
        # 这是一个简化的实现，实际应该使用SQL解析器
        pattern = r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return matches
    
    def add_limit_clause(self, sql: str) -> str:
        """为SQL添加LIMIT子句"""
        sql = sql.strip()
        if sql.endswith(';'):
            sql = sql[:-1]
        
        # 检查是否已有LIMIT
        if re.search(r'\bLIMIT\s+\d+', sql, re.IGNORECASE):
            # 已有LIMIT，检查是否超过最大值
            match = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
            if match:
                limit = int(match.group(1))
                if limit > self.max_rows:
                    sql = re.sub(
                        r'\bLIMIT\s+\d+', 
                        f'LIMIT {self.max_rows}', 
                        sql, 
                        flags=re.IGNORECASE
                    )
        else:
            # 没有LIMIT，添加一个
            sql += f" LIMIT {self.max_rows}"
        
        return sql
    
    def set_allowed_tables(self, prefixes: list):
        """设置允许访问的表前缀"""
        self.allowed_table_prefixes = prefixes
    
    def set_max_rows(self, max_rows: int):
        """设置最大返回行数"""
        self.max_rows = max_rows


