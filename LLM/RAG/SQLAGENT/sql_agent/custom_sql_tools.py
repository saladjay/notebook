"""
自定义 SQL Agent 的四个工具
可以添加额外的验证、日志记录、错误处理等功能
"""
from langchain.tools import BaseTool
from langchain_community.utilities import SQLDatabase
from typing import Optional, Type
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 关闭第三方库的详细日志（避免干扰）
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ============================================
# 工具 1: 自定义 sql_db_list_tables
# ============================================

class ListTablesInput(BaseModel):
    """sql_db_list_tables 的输入 schema"""
    # 这个工具不需要输入参数，但为了符合 LangChain 规范，定义一个空的 schema
    pass


class CustomSQLDatabaseListTables(BaseTool):
    """
    自定义的列出表名工具
    可以添加过滤、排序、日志等功能
    """
    name: str = "sql_db_list_tables"
    description: str = """
    Input is an empty string, output is a comma-separated list of tables in the database.
    
    使用此工具查看数据库中有哪些表可用。
    """
    args_schema: Type[BaseModel] = ListTablesInput
    
    db: SQLDatabase = Field(exclude=True)  # 数据库连接
    
    def _run(self, tool_input: str = "") -> str:
        """执行工具逻辑"""
        start_time = datetime.now()
        
        try:
            # 获取所有表名
            tables = self.db.get_usable_table_names()
            
            # 可以添加过滤逻辑
            # filtered_tables = [t for t in tables if not t.startswith('_')]
            
            result = ", ".join(tables)
            
            # 记录日志
            logger.info(f"列出表名: {result}")
            logger.info(f"耗时: {(datetime.now() - start_time).total_seconds():.3f}s")
            
            return result
            
        except Exception as e:
            error_msg = f" 获取表名失败: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _arun(self, tool_input: str = "") -> str:
        """异步版本（可选）"""
        return self._run(tool_input)


# ============================================
# 工具 2: 自定义 sql_db_schema
# ============================================

class SchemaInput(BaseModel):
    """sql_db_schema 的输入 schema"""
    table_names: str = Field(
        description="逗号分隔的表名列表，例如: 'table1, table2, table3'"
    )


class CustomSQLDatabaseSchema(BaseTool):
    """
    自定义的获取表结构工具
    可以控制返回的示例行数、格式化输出等
    """
    name: str = "sql_db_schema"
    description: str = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling sql_db_list_tables first!
    Example Input: table1, table2, table3
    
    使用此工具查看表的结构和示例数据。
    """
    args_schema: Type[BaseModel] = SchemaInput
    
    db: SQLDatabase = Field(exclude=True)
    sample_rows: int = Field(default=3, description="返回的示例行数")
    
    def _run(self, table_names: str) -> str:
        """执行工具逻辑"""
        start_time = datetime.now()
        
        try:
            # 清理输入
            tables = [t.strip() for t in table_names.split(",")]
            
            # 验证表是否存在
            valid_tables = self.db.get_usable_table_names()
            logger.info(f"可用的表: {valid_tables}")
            invalid_tables = [t for t in tables if t not in valid_tables]
            
            if invalid_tables:
                return f" 以下表不存在: {', '.join(invalid_tables)}\n可用的表: {', '.join(valid_tables)}"
            
            # 获取表结构
            result = self.db.get_table_info_no_throw(tables)
            
            # 可以添加自定义格式化
            # result = self._format_schema(result)
            
            # 记录日志
            logger.info(f"获取表结构: {table_names}")
            logger.info(f"耗时: {(datetime.now() - start_time).total_seconds():.3f}s")
            
            return result
            
        except Exception as e:
            error_msg = f" 获取表结构失败: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_schema(self, schema: str) -> str:
        """格式化 schema 输出（可选）"""
        # 可以添加额外的格式化逻辑
        return schema
    
    async def _arun(self, table_names: str) -> str:
        """异步版本（可选）"""
        return self._run(table_names)


# ============================================
# 工具 3: 自定义 sql_db_query（最重要）
# ============================================

class QueryInput(BaseModel):
    """sql_db_query 的输入 schema"""
    query: str = Field(
        description="完整的 SQL 查询语句，必须是语法正确的 SQL"
    )


class CustomSQLDatabaseQuery(BaseTool):
    """
    自定义的 SQL 查询执行工具
    可以添加：
    - SQL 注入防护
    - 查询结果限制
    - 执行时间限制
    - 详细的日志记录
    - 强制执行检查
    """
    name: str = "sql_db_query"
    description: str = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields.
    
    重要：你**必须**先使用 sql_db_query_checker 检查 SQL 语法，然后才能使用此工具！
    这是执行实际查询的工具，会返回真实的数据库结果。
    """
    args_schema: Type[BaseModel] = QueryInput
    
    db: SQLDatabase = Field(exclude=True)
    max_results: int = Field(default=100, description="最大返回行数")
    query_timeout: int = Field(default=10, description="查询超时时间（秒）")
    
    # 追踪工具调用历史
    call_history: list = Field(default_factory=list)
    
    def _run(self, query: str) -> str:
        """执行 SQL 查询"""
        start_time = datetime.now()
        
        # 记录调用
        call_record = {
            "timestamp": start_time.isoformat(),
            "query": query,
            "status": "pending"
        }
        
        try:
            # 1. 安全检查：禁止 DML 语句
            dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
            query_upper = query.upper()
            
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    error_msg = f"🚫 禁止执行 {keyword} 语句！只允许 SELECT 查询。"
                    call_record["status"] = "blocked"
                    call_record["error"] = error_msg
                    self.call_history.append(call_record)
                    logger.warning(error_msg)
                    return error_msg
            
            # 2. 添加 LIMIT 限制（如果查询中没有）
            # if "LIMIT" not in query_upper:
            #     query = f"{query.rstrip(';')} LIMIT {self.max_results}"
            #     logger.info(f"自动添加 LIMIT {self.max_results}")
            
            # 3. 执行查询
            logger.info(f"执行 SQL: {query}")
            result = self.db.run(query)
            
            # 4. 格式化结果
            if not result or result.strip() == "":
                result = "查询成功，但没有返回任何结果。"
            
            # 5. 记录成功
            execution_time = (datetime.now() - start_time).total_seconds()
            call_record["status"] = "success"
            call_record["result_length"] = len(str(result))
            call_record["execution_time"] = execution_time
            self.call_history.append(call_record)
            
            logger.info(f"查询成功")
            logger.info(f"返回结果长度: {len(str(result))} 字符")
            logger.info(f"执行时间: {execution_time:.3f}s")
            
            # 可以在结果前添加元信息
            # result = f"[执行时间: {execution_time:.3f}s]\n{result}"
            
            return result
            
        except Exception as e:
            error_msg = f" SQL 执行失败: {str(e)}\n\n请检查 SQL 语法是否正确，或使用 sql_db_schema 查看正确的表结构。"
            
            call_record["status"] = "error"
            call_record["error"] = str(e)
            self.call_history.append(call_record)
            
            logger.error(error_msg)
            return error_msg
    
    def get_statistics(self) -> dict:
        """获取查询统计信息"""
        total_calls = len(self.call_history)
        successful = len([c for c in self.call_history if c["status"] == "success"])
        errors = len([c for c in self.call_history if c["status"] == "error"])
        blocked = len([c for c in self.call_history if c["status"] == "blocked"])
        
        return {
            "total_calls": total_calls,
            "successful": successful,
            "errors": errors,
            "blocked": blocked,
            "success_rate": f"{successful/total_calls*100:.1f}%" if total_calls > 0 else "N/A"
        }
    
    async def _arun(self, query: str) -> str:
        """异步版本"""
        return self._run(query)


# ============================================
# 工具 4: 自定义 sql_db_query_checker
# ============================================

class QueryCheckerInput(BaseModel):
    """sql_db_query_checker 的输入 schema"""
    query: str = Field(
        description="需要检查的 SQL 查询语句"
    )


def check_sql_syntax(query: str, llm: Optional[any] = None) -> str:

    """检查 SQL 语法"""
    try:
        print("start to check:", query)
        #
        # 1. 基本格式检查
        query = query.strip()
        if not query:
            return " SQL 查询为空，请提供有效的 SQL 语句。"
        
        # 2. 安全检查
        dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return f" 检查失败：禁止使用 {keyword} 语句！\n\n只允许 SELECT 查询。", False
        
        # 3. SELECT 检查
        if not query_upper.startswith("SELECT"):
            return f" 检查失败：查询必须以 SELECT 开头。\n\n当前查询: {query[:100]}...", False
        
        # 4. 语法检查（简单版本）
        # 可以使用 sqlparse 库进行更详细的语法检查
        issues = []
        
        # 检查是否有未闭合的括号
        if query.count("(") != query.count(")"):
            issues.append("括号未配对")
        
        # 检查是否有未闭合的引号
        if query.count("'") % 2 != 0:
            issues.append("单引号未配对")
        if query.count('"') % 2 != 0:
            issues.append("双引号未配对")
        
        # 检查常见的 SQL 关键字拼写
        # 可以扩展更多检查规则
        
        if issues:
            return f" 检查失败：\n" + "\n".join(f"  - {issue}" for issue in issues), False
        

        
        # 返回优化后的 SQL（可以添加格式化）
        checked_query = query
        
        # 添加提示信息
        result = f"{checked_query}\n\n 语法检查通过！\n 下一步：请使用 sql_db_query 执行此查询以获取实际结果。"
        
        return result, True
        
    except Exception as e:
        error_msg = f" 检查过程出错: {str(e)}"
        logger.error(error_msg)
        return error_msg, False

class CustomSQLDatabaseQueryChecker(BaseTool):
    """
    自定义的 SQL 语法检查工具
    可以添加更严格的验证规则
    """
    name: str = "sql_db_query_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!
    
    重要：这个工具只检查 SQL 语法，不执行查询！
    检查通过后，你**必须**使用 sql_db_query 来执行查询并获取实际结果。
    
    输入: SQL 查询语句
    输出: 
    - 如果语法正确，返回检查后的 SQL（可能包含优化建议）
    - 如果有错误，返回错误信息和修改建议
    """
    args_schema: Type[BaseModel] = QueryCheckerInput
    
    db: SQLDatabase = Field(exclude=True)
    llm: Optional[any] = Field(default=None, exclude=True, description="用于智能检查的 LLM（可选）")
    
    def _run(self, query: str) -> str:
        """检查 SQL 语法"""
        start_time = datetime.now()
        result, flag = check_sql_syntax(query, None)
        if not flag:
            return result
        else:
            # 4. 使用 LLM 进行智能检查（如果配置了）
            try:
                if self.llm:
                    llm_check_result = self._llm_check(query)
                    if llm_check_result:
                        return llm_check_result
                
                # 6. 检查通过
                logger.info(f"SQL 语法检查通过")
                logger.info(f"检查耗时: {(datetime.now() - start_time).total_seconds():.3f}s")
                
                # 返回优化后的 SQL（可以添加格式化）
                checked_query = query
                
                # 添加提示信息
                result = f"{checked_query}\n\n 语法检查通过！\n 下一步：请使用 sql_db_query 执行此查询以获取实际结果。"
                
                return result
            
            except Exception as e:
                error_msg = f" 检查过程出错: {str(e)}"
                logger.error(error_msg)
                return error_msg
    
    def _llm_check(self, query: str) -> Optional[str]:
        """使用 LLM 进行智能语法检查（可选）"""
        # 可以调用 LLM 来检查更复杂的语法问题
        # 这里留空，可以根据需要实现
        return None
    
    async def _arun(self, query: str) -> str:
        """异步版本"""
        return self._run(query)


# ============================================
# 工具包：整合所有自定义工具
# ============================================

class CustomSQLDatabaseToolkit:
    """
    自定义 SQL 工具包
    替代 LangChain 的默认 SQLDatabaseToolkit
    """
    
    def __init__(
        self, 
        db: SQLDatabase, 
        llm: Optional[any] = None,
        sample_rows: int = 3,
        max_results: int = 100
    ):
        """
        初始化自定义工具包
        
        Args:
            db: SQLDatabase 实例
            llm: 用于智能检查的 LLM（可选）
            sample_rows: schema 工具返回的示例行数
            max_results: query 工具返回的最大结果数
        """
        self.db = db
        self.llm = llm
        self.sample_rows = sample_rows
        self.max_results = max_results
    
    def get_tools(self) -> list:
        """
        获取所有自定义工具
        返回顺序很重要，影响 Agent 的工具选择
        """
        tools = [
            CustomSQLDatabaseListTables(db=self.db),
            CustomSQLDatabaseSchema(db=self.db, sample_rows=self.sample_rows),
            CustomSQLDatabaseQueryChecker(db=self.db, llm=self.llm),
            CustomSQLDatabaseQuery(db=self.db, max_results=self.max_results),
        ]
        
        return tools


# ============================================
# 使用示例
# ============================================

def create_agent_with_custom_tools():
    """
    创建使用自定义工具的 SQL Agent
    """
    from langchain_community.utilities import SQLDatabase
    from langchain_community.llms import Ollama
    from langchain.agents import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains import LLMChain
    from config import get_config
    
    config = get_config()
    
    # 1. 初始化数据库
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    # 2. 初始化 LLM
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    # 3. 创建自定义工具包
    toolkit = CustomSQLDatabaseToolkit(
        db=db,
        llm=llm,
        sample_rows=3,
        max_results=50
    )
    tools = toolkit.get_tools()
    
    print(f" 创建了 {len(tools)} 个自定义工具:")
    for tool in tools:
        print(f"  - {tool.name}")
    
    # 4. 创建自定义 Prompt
    prefix = """你是一个专业的 SQL 数据库助手。

严格执行规则：
1. 先使用 sql_db_list_tables 查看可用的表
2. 使用 sql_db_schema 查看相关表的结构
3. 编写 SQL 查询后，**必须**先用 sql_db_query_checker 检查
4. 检查通过后，**必须**用 sql_db_query 执行查询
5. **只有看到 sql_db_query 的实际结果后**，才能给出 Final Answer

禁止的行为：
 在没有执行 sql_db_query 的情况下猜测答案
 跳过 sql_db_query_checker 直接执行查询
 使用 DML 语句（INSERT, UPDATE, DELETE 等）

你有以下工具可用："""

    suffix = """开始！记住要严格按照流程执行。

Question: {input}
Thought: 我需要查看数据库中有哪些表，然后查看相关表的结构。
{agent_scratchpad}"""

    # 5. 创建 Prompt
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
    )
    
    # 6. 创建 LLM Chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # 7. 创建 Agent
    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=[tool.name for tool in tools],
    )
    
    # 8. 创建 Agent Executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
    )
    
    return agent_executor, tools


def demo_custom_tools():
    """演示自定义工具的使用"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🛠️  自定义 SQL Agent 工具演示                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建 Agent
    agent, tools = create_agent_with_custom_tools()
    
    # 测试查询
    test_questions = [
        "数据库里有哪些表？",
        "Image 表有多少条记录？",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"测试 {i}: {question}")
        print(f"{'='*80}\n")
        
        try:
            result = agent.invoke({"input": question})
            print(f"\n✅ 最终答案: {result.get('output', result)}\n")
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}\n")
    
    # 显示查询统计
    query_tool = [t for t in tools if t.name == "sql_db_query"][0]
    if isinstance(query_tool, CustomSQLDatabaseQuery):
        print("\n" + "="*80)
        print("📊 查询统计")
        print("="*80)
        stats = query_tool.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*80)


if __name__ == '__main__':
    
    sql_list = ["SELECT Id FROM DataSet WHERE Name LIKE '2027_01_31_8'", "SELECT Id FROM DataSet WHERE Name = '2028_07_29' OR Name = '2027_07_05_2'"]
    for sql in sql_list:
        print(check_sql_syntax(sql))
    
