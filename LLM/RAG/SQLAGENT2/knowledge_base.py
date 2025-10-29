"""
知识库模块
存储数据库schema、常见查询模式、业务规则等
"""
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.engine import Engine


class KnowledgeBase:
    """知识库"""
    
    def __init__(self, database_uri: str):
        """
        初始化知识库
        
        Args:
            database_uri: 数据库连接URI
        """
        self.engine: Engine = create_engine(database_uri)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        
        # 业务规则
        self.business_rules: Dict[str, str] = {}
        
        # 常见查询模式
        self.query_patterns: List[Dict[str, str]] = []
        
        # 表描述
        self.table_descriptions: Dict[str, str] = {}
        
        # 列描述
        self.column_descriptions: Dict[str, Dict[str, str]] = {}
    
    def get_database_schema(self) -> str:
        """获取数据库schema的文本描述"""
        inspector = inspect(self.engine)
        schema_text = "数据库Schema信息：\n\n"
        
        for table_name in inspector.get_table_names():
            # 表名和描述
            table_desc = self.table_descriptions.get(table_name, "")
            schema_text += f"表名: {table_name}"
            if table_desc:
                schema_text += f" - {table_desc}"
            schema_text += "\n"
            
            # 列信息
            columns = inspector.get_columns(table_name)
            for column in columns:
                col_name = column['name']
                col_type = str(column['type'])
                col_desc = self.column_descriptions.get(table_name, {}).get(col_name, "")
                
                schema_text += f"  - {col_name} ({col_type})"
                if col_desc:
                    schema_text += f": {col_desc}"
                schema_text += "\n"
            
            # 主键
            pk = inspector.get_pk_constraint(table_name)
            if pk and pk['constrained_columns']:
                schema_text += f"  主键: {', '.join(pk['constrained_columns'])}\n"
            
            # 外键
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                for fk in fks:
                    schema_text += f"  外键: {', '.join(fk['constrained_columns'])} -> "
                    schema_text += f"{fk['referred_table']}.{', '.join(fk['referred_columns'])}\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def get_table_names(self) -> List[str]:
        """获取所有表名"""
        inspector = inspect(self.engine)
        return inspector.get_table_names()
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """获取表的详细信息"""
        inspector = inspect(self.engine)
        
        return {
            "name": table_name,
            "description": self.table_descriptions.get(table_name, ""),
            "columns": inspector.get_columns(table_name),
            "primary_key": inspector.get_pk_constraint(table_name),
            "foreign_keys": inspector.get_foreign_keys(table_name),
            "indexes": inspector.get_indexes(table_name)
        }
    
    def add_table_description(self, table_name: str, description: str):
        """添加表描述"""
        self.table_descriptions[table_name] = description
    
    def add_column_description(self, table_name: str, column_name: str, description: str):
        """添加列描述"""
        if table_name not in self.column_descriptions:
            self.column_descriptions[table_name] = {}
        self.column_descriptions[table_name][column_name] = description
    
    def add_business_rule(self, rule_name: str, rule_description: str):
        """添加业务规则"""
        self.business_rules[rule_name] = rule_description
    
    def get_business_rules(self) -> str:
        """获取所有业务规则"""
        if not self.business_rules:
            return ""
        
        rules_text = "业务规则：\n"
        for name, description in self.business_rules.items():
            rules_text += f"- {name}: {description}\n"
        return rules_text
    
    def add_query_pattern(self, pattern_name: str, description: str, example_sql: str):
        """添加常见查询模式"""
        self.query_patterns.append({
            "name": pattern_name,
            "description": description,
            "example_sql": example_sql
        })
    
    def get_query_patterns(self) -> str:
        """获取常见查询模式"""
        if not self.query_patterns:
            return ""
        
        patterns_text = "常见查询模式：\n"
        for pattern in self.query_patterns:
            patterns_text += f"\n模式: {pattern['name']}\n"
            patterns_text += f"描述: {pattern['description']}\n"
            patterns_text += f"示例SQL: {pattern['example_sql']}\n"
        return patterns_text
    
    def get_context_for_question(self, question: str) -> str:
        """根据问题获取相关上下文"""
        context = self.get_database_schema()
        
        business_rules = self.get_business_rules()
        if business_rules:
            context += "\n" + business_rules
        
        query_patterns = self.get_query_patterns()
        if query_patterns:
            context += "\n" + query_patterns
        
        return context

