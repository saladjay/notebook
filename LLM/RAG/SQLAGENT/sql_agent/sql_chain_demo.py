"""
使用LangChain SQLDatabaseChain的简化版本
适合快速查询和简单场景
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

def main():
    print("=" * 60)
    print("🚀 SQL Chain Demo - 简化版")
    print("=" * 60)
    
    # 初始化模型和数据库
    print("\n📡 初始化LLM和数据库连接...")
    llm = Ollama(model="llama3.2", temperature=0)
    db = SQLDatabase.from_uri("sqlite:///company.db")
    print("✅ 初始化完成！")
    
    # 创建SQL查询链
    print("\n🔗 创建SQL查询链...")
    query_chain = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    
    # 示例查询
    queries = [
        "列出所有员工的姓名和部门",
        "技术部有多少人？",
        "工资最高的员工是谁？"
    ]
    
    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"🔍 查询: {query}")
        print("=" * 60)
        
        try:
            # 生成SQL
            sql_query = query_chain.invoke({"question": query})
            print(f"\n📝 生成的SQL:\n{sql_query}")
            
            # 执行SQL
            result = execute_query.invoke(sql_query)
            print(f"\n📊 查询结果:\n{result}")
            
        except Exception as e:
            print(f"❌ 错误: {str(e)}")

if __name__ == '__main__':
    main()

