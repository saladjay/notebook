"""
SQL Agent with Few-Shot Examples
为 LLM 提供示例来提高查询准确率
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from config import get_config
from callbacks import ChainLoggingCallback, LLMLoggingCallback

# ============================================
# 方法1: 自定义 PREFIX（最简单）
# ============================================

def create_agent_with_examples_v1():
    """
    方法1: 在 PREFIX 中添加示例
    适合：简单场景，少量示例
    """
    config = get_config()
    
    # 初始化
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    # 自定义 PREFIX，添加示例
    custom_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run.

Here are some examples of questions and their corresponding SQL queries:

Example 1:
Question: How many images are in the train dataset?
SQL Query: SELECT COUNT(*) FROM Image WHERE DataSet_id = (SELECT Id FROM DataSet WHERE Name = 'train')

Example 2:
Question: 在这个项目里有什么标签？
SQL Query: SELECT Name FROM LabelClass

Example 3:
Question: What is the sizes of all labels?
SQL Query: SELECT CAST(json_extract(Region, '$.h') AS REAL) * CAST(json_extract(Region, '$.w') AS REAL) AS area FROM Label

Now, please answer the following question following the same pattern.

Unless the user specifies a specific number of examples they wish to obtain, 
always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples.
Never query for all the columns from a specific table, only ask for the relevant columns.

You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""
    
    # 创建 Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=custom_prefix,  # 使用自定义 PREFIX
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[ChainLoggingCallback()]
    )
    
    return agent


# ============================================
# 方法2: 使用 FewShotPromptTemplate（推荐）
# ============================================

def create_agent_with_examples_v2():
    """
    方法2: 使用 FewShotPromptTemplate
    适合：较多示例，需要动态选择示例
    """
    config = get_config()
    
    # 初始化
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    # 定义示例
    examples = [
        {
            "input": "How many images are in the train dataset?",
            "query": "SELECT COUNT(*) FROM Image WHERE DataSet_id = (SELECT Id FROM DataSet WHERE Name = 'train')"
        },
        {
            "input": "在这个项目里有什么标签？",
            "query": "SELECT DISTINCT label FROM dataset ORDER BY label"
        },
        {
            "input": "What are the sizes of all labels?",
            "query": "SELECT CAST(json_extract(Region, '$.h') AS REAL) * CAST(json_extract(Region, '$.w') AS REAL) AS area FROM Label"
        },
        # {
        #     "input": "Show me the first 5 images from validation set",
        #     "query": "SELECT * FROM dataset WHERE split='validation' LIMIT 5"
        # },
    ]
    
    # 创建示例模板
    example_template = """
Question: {input}
SQL Query: {query}
"""
    
    example_prompt = PromptTemplate(
        input_variables=["input", "query"],
        template=example_template
    )
    
    # 创建 Few-Shot Prompt
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""You are a SQL expert. Here are some example questions and their SQL queries:""",
        suffix="""Now answer this question:
Question: {input}
SQL Query:""",
        input_variables=["input"]
    )
    
    # 构建完整的 PREFIX
    examples_text = "\n".join([
        f"Question: {ex['input']}\nSQL Query: {ex['query']}"
        for ex in examples
    ])
    
    custom_prefix = f"""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run.

Here are some examples to help you understand the database structure and query patterns:

{examples_text}

Please follow these examples when creating queries for new questions.

Unless the user specifies a specific number of examples they wish to obtain, 
always limit your query to at most 5 results.
Never query for all the columns from a specific table, only ask for the relevant columns.

You have access to tools for interacting with the database.
Only use the below tools.
You MUST double check your query before executing it.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""
    
    # 创建 Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=custom_prefix,
        verbose=True,
        handle_parsing_errors=True,
        callbacks=[LLMLoggingCallback(log_file="test_working.log")]
    )
    
    return agent


# ============================================
# 方法3: 从文件加载示例（灵活）
# ============================================

def load_examples_from_file(filename="sql_examples.txt"):
    """
    从文件加载示例
    文件格式：
    Q: 问题1
    A: SQL查询1
    ---
    Q: 问题2
    A: SQL查询2
    """
    import os
    
    if not os.path.exists(filename):
        # 创建示例文件
        default_examples = """Q: How many images are in the train dataset?
A: SELECT COUNT(*) FROM dataset WHERE split='train'
---
Q: List all unique labels in the dataset
A: SELECT DISTINCT label FROM dataset ORDER BY label
---
Q: What is the average size of images?
A: SELECT AVG(width * height) as avg_size FROM images
---
Q: Show me images with label 'cat'
A: SELECT * FROM dataset WHERE label='cat' LIMIT 5
"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(default_examples)
        print(f"✅ 已创建示例文件: {filename}")
    
    examples = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 解析示例
    example_blocks = content.strip().split('---')
    for block in example_blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            question = lines[0].replace('Q:', '').strip()
            answer = lines[1].replace('A:', '').strip()
            examples.append({
                'input': question,
                'query': answer
            })
    
    return examples


def create_agent_with_file_examples():
    """
    方法3: 从文件加载示例
    适合：需要经常更新示例，团队协作
    """
    config = get_config()
    
    # 初始化
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    # 从文件加载示例
    examples = load_examples_from_file()
    
    print(f"📚 已加载 {len(examples)} 个示例")
    
    # 构建 PREFIX
    examples_text = "\n\n".join([
        f"Example {i+1}:\nQuestion: {ex['input']}\nSQL Query: {ex['query']}"
        for i, ex in enumerate(examples)
    ])
    
    custom_prefix = f"""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run.

Here are some examples to guide you:

{examples_text}

Please follow these examples when creating queries.

Unless the user specifies a specific number of examples they wish to obtain, 
always limit your query to at most 5 results.

You have access to tools for interacting with the database.
Only use the below tools.
You MUST double check your query before executing it.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""
    
    # 创建 Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=custom_prefix,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent


# ============================================
# 方法4: 动态示例（根据问题选择相关示例）
# ============================================

def create_agent_with_dynamic_examples():
    """
    方法4: 根据问题动态选择最相关的示例
    适合：大量示例，需要智能选择
    """
    config = get_config()
    
    # 初始化
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    # 分类的示例库
    examples_by_category = {
        "count": [
            ("How many images are in the train dataset?", 
             "SELECT COUNT(*) FROM dataset WHERE split='train'"),
            ("Count all records", 
             "SELECT COUNT(*) FROM dataset"),
        ],
        "filter": [
            ("Show images with label 'cat'", 
             "SELECT * FROM dataset WHERE label='cat' LIMIT 5"),
            ("Get validation set data", 
             "SELECT * FROM dataset WHERE split='validation' LIMIT 5"),
        ],
        "aggregate": [
            ("What is the average image size?", 
             "SELECT AVG(width * height) FROM images"),
            ("Sum of all values", 
             "SELECT SUM(column_name) FROM table_name"),
        ],
        "distinct": [
            ("List all unique labels", 
             "SELECT DISTINCT label FROM dataset ORDER BY label"),
            ("Get unique splits", 
             "SELECT DISTINCT split FROM dataset"),
        ]
    }
    
    # 根据关键词选择示例
    def select_examples(question: str, max_examples: int = 3):
        """根据问题选择最相关的示例"""
        question_lower = question.lower()
        selected = []
        
        # 关键词映射
        keywords = {
            "count": ["count", "how many", "number of"],
            "filter": ["show", "get", "where", "filter"],
            "aggregate": ["average", "sum", "max", "min", "avg"],
            "distinct": ["unique", "distinct", "different"]
        }
        
        # 找到相关类别
        for category, words in keywords.items():
            if any(word in question_lower for word in words):
                if category in examples_by_category:
                    selected.extend(examples_by_category[category][:2])
        
        # 如果没找到，返回通用示例
        if not selected:
            selected = list(examples_by_category["count"])[:2]
        
        return selected[:max_examples]
    
    # 示例：为特定问题选择示例
    question = "How many images are there in the train dataset?"
    relevant_examples = select_examples(question)
    
    examples_text = "\n\n".join([
        f"Example:\nQuestion: {q}\nSQL Query: {sql}"
        for q, sql in relevant_examples
    ])
    
    custom_prefix = f"""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run.

Here are some relevant examples:

{examples_text}

Follow these examples to create your query.

Unless the user specifies a specific number of examples they wish to obtain, 
always limit your query to at most 5 results.

You have access to tools for interacting with the database.
You MUST double check your query before executing it.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""
    
    # 创建 Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prefix=custom_prefix,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent


# ============================================
# 测试和对比
# ============================================

def test_agents():
    """测试不同方法的效果"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🧪 测试带示例的 SQL Agent                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    test_questions = [
        "在train数据集里有几张图片?",
        "给出五张label高最大的图片信息",
        "这个项目有几个数据集?",
    ]
    
    print("\n选择测试方法:")
    print("  [1] 方法1: 自定义 PREFIX（简单）")
    print("  [2] 方法2: FewShotPromptTemplate（推荐）")
    print("  [3] 方法3: 从文件加载示例（灵活）")
    print("  [4] 方法4: 动态选择示例（智能）")
    print("  [0] 退出")
    
    choice = input("\n请选择 (0-4): ").strip()
    
    if choice == '0':
        print("👋 再见！")
        return
    
    # 创建 Agent
    print(f"\n{'='*80}")
    print(f"创建 Agent...")
    print(f"{'='*80}")
    
    if choice == '1':
        agent = create_agent_with_examples_v1()
    elif choice == '2':
        agent = create_agent_with_examples_v2()
    elif choice == '3':
        agent = create_agent_with_file_examples()
    elif choice == '4':
        agent = create_agent_with_dynamic_examples()
    else:
        print("❌ 无效选项")
        return
    callbacks = [LLMLoggingCallback(json_file="test_working.json", log_to_file=False, log_to_console=False, save_json=True)]
    # 测试查询
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"测试 {i}/{len(test_questions)}: {question}")
        print(f"{'='*80}")
        
        try:
            result = agent.invoke({"input": question}, config={"callbacks": callbacks})
            print(f"\n✅ 答案: {result.get('output', result)}")
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
    
    print(f"\n{'='*80}")
    print("✅ 测试完成")
    print(f"{'='*80}")


def main():
    """主函数"""
    test_agents()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

