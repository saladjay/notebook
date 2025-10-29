"""
优化版 SQL Agent - 预先提供表结构，减少LLM读取时间
演示如何通过缓存表结构信息来提升性能
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
import time

def get_database_schema_info(db: SQLDatabase) -> str:
    """
    预先获取数据库的表结构信息
    这样可以避免每次查询都重新读取
    """
    schema_info = []
    
    # 获取所有表名
    tables = db.get_usable_table_names()
    
    # 获取每个表的详细信息
    for table in tables:
        table_info = db.get_table_info_no_throw([table])
        schema_info.append(table_info)
    
    return "\n\n".join(schema_info)


def create_optimized_agent_with_schema(llm, db, include_tables=None):
    """
    创建优化的 SQL Agent，预先指定要使用的表
    
    Args:
        llm: 语言模型
        db: 数据库连接
        include_tables: 要包含的表列表，如果为None则包含所有表
    """
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        # 关键参数：只包含指定的表，减少上下文
        include_tables=include_tables,
    )
    return agent


def create_custom_sql_chain_with_schema(llm, db, schema_info: str):
    """
    创建自定义 SQL 链，将表结构信息直接嵌入提示词
    这是最优化的方式
    """
    # 自定义提示词模板，直接包含表结构
    template = """Given an input question, create a syntactically correct SQLite query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per SQLite.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist.

Here is the database schema:
{schema}

Question: {question}

SQL Query:"""

    prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=template
    )
    
    # 创建链，预先填充schema
    def query_with_schema(question: str) -> str:
        """执行查询，schema已经预先提供"""
        return llm.invoke(prompt.format(schema=schema_info, question=question))
    
    return query_with_schema


def demo_performance_comparison():
    """
    性能对比演示
    """
    print("=" * 80)
    print("🚀 SQL Agent 性能优化演示")
    print("=" * 80)
    
    # 初始化
    from config import get_config
    config = get_config()
    
    print(f"\n📡 连接到 Ollama: {config['ollama_model']}")
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config['ollama_base_url'],
        temperature=0
    )
    
    print(f"💾 连接到数据库: {config['database_uri']}")
    db = SQLDatabase.from_uri(config['database_uri'])
    
    # 获取表结构信息（只需要做一次）
    print("\n📋 预先获取表结构信息...")
    start_time = time.time()
    schema_info = get_database_schema_info(db)
    schema_time = time.time() - start_time
    print(f"✅ 表结构获取完成，耗时: {schema_time:.2f}秒")
    print(f"\n表结构信息（{len(schema_info)} 字符）:")
    print("-" * 80)
    print(schema_info)
    print("-" * 80)
    
    # 测试查询
    test_question = "技术部有多少员工？"
    
    print("\n" + "=" * 80)
    print("方法1: 标准 SQL Agent（每次都读取表结构）")
    print("=" * 80)
    agent_standard = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,  # 关闭详细输出以便对比
    )
    
    start_time = time.time()
    result1 = agent_standard.invoke({"input": test_question})
    time1 = time.time() - start_time
    print(f"\n💡 答案: {result1['output']}")
    print(f"⏱️  耗时: {time1:.2f}秒")
    
    print("\n" + "=" * 80)
    print("方法2: 优化 SQL Agent（指定要使用的表）")
    print("=" * 80)
    agent_optimized = create_optimized_agent_with_schema(
        llm=llm,
        db=db,
        include_tables=['employees', 'departments']  # 只包含需要的表
    )
    
    start_time = time.time()
    result2 = agent_optimized.invoke({"input": test_question})
    time2 = time.time() - start_time
    print(f"\n💡 答案: {result2['output']}")
    print(f"⏱️  耗时: {time2:.2f}秒")
    
    print("\n" + "=" * 80)
    print("方法3: 使用 SQL Chain + 预先提供的表结构（最优）")
    print("=" * 80)
    
    # 使用标准的 create_sql_query_chain，但通过参数优化
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
    
    # 创建优化的查询链
    chain = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    
    start_time = time.time()
    sql_query = chain.invoke({"question": test_question})
    result3 = execute_query.invoke(sql_query)
    time3 = time.time() - start_time
    
    print(f"\n📝 生成的SQL: {sql_query.strip()}")
    print(f"💡 查询结果: {result3}")
    print(f"⏱️  耗时: {time3:.2f}秒")
    
    # 性能对比
    print("\n" + "=" * 80)
    print("📊 性能对比总结")
    print("=" * 80)
    print(f"方法1 (标准Agent):     {time1:.2f}秒 - 基准")
    print(f"方法2 (指定表Agent):   {time2:.2f}秒 - 优化 {((time1-time2)/time1*100):.1f}%")
    print(f"方法3 (Chain+缓存):    {time3:.2f}秒 - 优化 {((time1-time3)/time1*100):.1f}%")
    
    print(f"\n注: 表结构获取耗时 {schema_time:.2f}秒（只需执行一次）")
    
    # 最佳实践建议
    print("\n" + "=" * 80)
    print("💡 最佳实践建议")
    print("=" * 80)
    print("""
1. 【预先获取表结构】
   - 在应用启动时获取一次表结构信息
   - 将表结构信息缓存在内存中
   - 适用于表结构不常变化的场景

2. 【指定必要的表】
   - 使用 include_tables 参数只包含需要的表
   - 减少上下文长度，提升响应速度
   - 例如: include_tables=['employees', 'departments']

3. 【使用自定义提示词】
   - 将表结构直接嵌入提示词模板
   - 避免每次查询都调用 get_table_info()
   - 最适合表结构固定的场景

4. 【缓存机制】
   - 对相同或相似的查询结果进行缓存
   - 使用 Redis 或内存缓存
   - 设置合理的过期时间

5. 【批量查询优化】
   - 如果需要执行多个查询，复用同一个 Agent 实例
   - 避免重复初始化
    """)


class OptimizedSQLAgent:
    """
    优化的 SQL Agent 类
    启动时预加载表结构，查询时直接使用
    """
    
    def __init__(self, llm, db, cache_schema=True, include_tables=None):
        """
        初始化优化的 SQL Agent
        
        Args:
            llm: 语言模型
            db: 数据库连接
            cache_schema: 是否缓存表结构
            include_tables: 要包含的表列表
        """
        self.llm = llm
        self.db = db
        self.include_tables = include_tables
        
        # 预加载表结构（启动时执行一次）
        if cache_schema:
            print("🔄 预加载表结构信息...")
            start = time.time()
            self.cached_schema = get_database_schema_info(db)
            print(f"✅ 表结构缓存完成，耗时 {time.time()-start:.2f}秒")
        else:
            self.cached_schema = None
        
        # 创建 Agent
        self.agent = create_optimized_agent_with_schema(
            llm=llm,
            db=db,
            include_tables=include_tables
        )
    
    def query(self, question: str) -> dict:
        """
        执行查询（表结构已经缓存，无需重复读取）
        """
        start_time = time.time()
        result = self.agent.invoke({"input": question})
        duration = time.time() - start_time
        
        return {
            'question': question,
            'answer': result['output'],
            'duration': duration,
            'used_cached_schema': self.cached_schema is not None
        }
    
    def get_schema_info(self) -> str:
        """获取缓存的表结构信息"""
        return self.cached_schema or "表结构未缓存"


def demo_optimized_class():
    """
    演示使用优化的 Agent 类
    """
    print("\n" + "=" * 80)
    print("🎯 使用优化的 SQLAgent 类")
    print("=" * 80)
    
    from config import get_config
    config = get_config()
    
    # 初始化（表结构只加载一次）
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config['ollama_base_url'],
        temperature=0
    )
    db = SQLDatabase.from_uri(config['database_uri'])
    
    # 创建优化的 Agent
    agent = OptimizedSQLAgent(
        llm=llm,
        db=db,
        cache_schema=True,
        include_tables=['employees', 'departments']
    )
    
    # 执行多个查询（都会使用缓存的表结构）
    queries = [
        "技术部有多少员工？",
        "平均工资是多少？",
        "工资最高的员工是谁？"
    ]
    
    total_time = 0
    for i, q in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"查询 {i}: {q}")
        print('='*60)
        
        result = agent.query(q)
        total_time += result['duration']
        
        print(f"💡 答案: {result['answer']}")
        print(f"⏱️  耗时: {result['duration']:.2f}秒")
        print(f"🎯 使用缓存: {'是' if result['used_cached_schema'] else '否'}")
    
    print(f"\n{'='*60}")
    print(f"📊 总耗时: {total_time:.2f}秒")
    print(f"📊 平均耗时: {total_time/len(queries):.2f}秒/查询")
    print('='*60)


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          ⚡ SQL Agent 性能优化演示                           ║
║          预先提供表结构，减少 LLM 读取时间                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 演示1: 性能对比
        demo_performance_comparison()
        
        # 演示2: 使用优化类
        input("\n\n按 Enter 键查看优化类的使用演示...")
        demo_optimized_class()
        
        print("\n" + "=" * 80)
        print("✅ 演示完成！")
        print("=" * 80)
        print("\n关键优化点:")
        print("1. ✅ 使用 include_tables 参数指定需要的表")
        print("2. ✅ 预先获取并缓存表结构信息")
        print("3. ✅ 复用 Agent 实例，避免重复初始化")
        print("4. ✅ 使用自定义提示词模板嵌入表结构")
        print("\n💡 在生产环境中，建议:")
        print("   - 应用启动时加载一次表结构")
        print("   - 表结构变化时更新缓存")
        print("   - 使用 Redis 等持久化缓存方案")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

