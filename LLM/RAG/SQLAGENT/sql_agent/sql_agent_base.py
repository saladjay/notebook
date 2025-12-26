"""
使用LangChain连接Ollama模型，读取和查询SQLite数据库的demo
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from config import get_config
import logging
import os
from db_utils import LoggingSQLDatabase
from langchain.callbacks.base import BaseCallbackHandler
from models.ChatModel import _get_bailian_llm, _get_ollama_llm
from llm_wrapper import LLMWrapper, Qwen3ReActInterceptor

class SQLAgentBase:
    def __init__(self, model_name: str = None, database_uri: str = None):
        config = get_config()
        self.model_name = model_name or config.get("ollama_model")
        self.database_uri = database_uri or config.get("database_uri")
        
        # 修正数据库URI格式（自动处理常见错误）
        self.database_uri = self._fix_sqlite_uri(self.database_uri)
        print(f"📋 使用数据库URI: {self.database_uri}")
        
        # 初始化LLM
        llm_kwargs = {
            "model": self.model_name,
            "temperature": config.get("temperature", 0),
        }
        if config.get("ollama_base_url"):
            llm_kwargs["base_url"] = config.get("ollama_base_url")
        self.llm = _get_bailian_llm()

        self.llm = LLMWrapper(self.llm, 
            interceptors=[Qwen3ReActInterceptor(jsonl_file="./generate_data/qwen3_react_history_20251103.jsonl")])


        # self.llm = _get_bailian_llm()
        
        # 初始化数据库连接
        self.db = LoggingSQLDatabase.from_uri(self.database_uri)
        
        # 测试 LLM 连接与生成能力
        try:
            response = self.llm.invoke("你是谁？请用一句话自我介绍。")
            print(f"✅ LLM 测试成功")
        except Exception as e:
            print(f"❌ LLM 调用失败: {str(e)}")
            raise
        
        # 测试数据库连接
        try:
            tables = self.db.get_usable_table_names()
            print(f"✅ 数据库连接成功！")
        except Exception as e:
            print(f"❌ 数据库连接失败: {str(e)}")
            raise
        
        # 创建Agent
        self.agent = self.create_agent()
    
    @staticmethod
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
    
    def create_agent(self):
        """子类需要实现这个方法"""
        pass

    def query(self, question: str, callback: list = None):
        """执行查询"""
        if callback:
            return self.agent.invoke({"input": question}, config={"callbacks":callback})
        else:
            return self.agent.invoke({"input": question})

    def _setup_logging(self):
        """设置日志"""
        config = get_config()
        if config.get('enable_logging'):
            logging.basicConfig(
                filename=config.get('log_file'),
                level=getattr(logging, config.get('log_level', 'INFO')),
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("=" * 60)
            self.logger.info("SQL Agent 启动")
            self.logger.info("=" * 60)
        else:
            self.logger = None
