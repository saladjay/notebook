"""
查看 LangChain SQL Agent 的默认 PREFIX 和 SUFFIX
"""
try:
    from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
    
    print("=" * 80)
    print("LangChain SQL Agent 默认 PREFIX")
    print("=" * 80)
    print(SQL_PREFIX)
    print("\n")
    
    print("=" * 80)
    print("LangChain SQL Agent 默认 SUFFIX")
    print("=" * 80)
    print(SQL_SUFFIX)
    
except ImportError as e:
    print(f"无法导入模块: {e}")
    print("\n尝试从在线源码查看...")
    
    # 显示 LangChain 默认的 SQL_PREFIX（从源码复制）
    SQL_PREFIX_DEFAULT = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer."""

    SQL_SUFFIX_DEFAULT = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""

    print("=" * 80)
    print("LangChain SQL Agent 默认 PREFIX（从源码获取）")
    print("=" * 80)
    print(SQL_PREFIX_DEFAULT)
    print("\n")
    
    print("=" * 80)
    print("LangChain SQL Agent 默认 SUFFIX（从源码获取）")
    print("=" * 80)
    print(SQL_SUFFIX_DEFAULT)

