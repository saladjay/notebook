"""
使用LangChain连接Ollama模型，读取和查询SQLite数据库的demo
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor

from sql_agent.sql_agent_base import SQLAgentBase
from sql_agent.qwen_react_parser import get_qwen_react_parser
prefix =  """你是一个 SQL 数据库交互代理。
    
关键规则：
1. 使用 sql_db_query_checker 检查 SQL 后，**必须**使用 sql_db_query 执行查询
2. **绝对不能**在没有实际执行查询的情况下给出答案
3. Final Answer 必须基于 sql_db_query 的实际执行结果

执行流程：
- sql_db_list_tables → sql_db_schema → sql_db_query_checker → **sql_db_query** → Final Answer
"""

class SQLAgentDemo(SQLAgentBase):
    def __init__(self, model_name: str = None, database_uri: str = None):
        super().__init__(model_name, database_uri)
    
    def create_agent(self):
        return create_sql_agent(
            llm=self.llm,  # ✅ 现在 LLMWrapper 继承自 BaseLanguageModel，可以直接使用
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            output_parser=get_qwen_react_parser(),
            prefix=prefix)


