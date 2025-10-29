"""
专门适配Qwen3模型的SQL Agent
支持Qwen3的思考模式(<think>标签)
"""
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl import prompt as react_prompt
from sql_agent.sql_agent_base import SQLAgentBase
from sql_agent.qwen_react_parser import get_qwen_react_parser
from llm_wrapper import LLMWrapper, Qwen3ReActInterceptor

class SQLAgentQwen(SQLAgentBase):
    """适配Qwen3思考模式的SQL Agent"""
    
    def __init__(self, model_name: str = None, database_uri: str = None):
        super().__init__(model_name, database_uri)
    
    def create_agent(self):
        """
        创建Agent，使用自定义的Qwen解析器
        
        方法1：使用handle_parsing_errors=True让Agent自动处理解析错误
        方法2：使用自定义解析器（更优雅但需要手动构建Agent）
        """
        # 方法1：简单方案 - 使用handle_parsing_errors
        # 这会让Agent在遇到解析错误时自动重试，并将错误信息反馈给LLM
        return create_sql_agent(
            llm=self.llm,  # ✅ 现在 LLMWrapper 继承自 BaseLanguageModel，可以直接使用
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,  # 关键：自动处理解析错误
            max_iterations=15,  # 增加迭代次数，因为可能需要多次重试
            max_execution_time=None,
            early_stopping_method="generate",
            # 添加Agent参数来引导模型输出正确格式
            agent_executor_kwargs={
                "handle_parsing_errors": "Check your output and make sure it conforms to the format instructions. Do not use <think> tags."
            }
        )


single_row_sample = """
DataSet: "Id": 1, "Name": "Dataset1", "Additional_data": b'{"locked": true}'
Image: "Id": 1, "Path": "D:/data/2025/01/01/image1.jpg", "DataSet_id": 1, "Tag_id": 1, "Additional_data": b'NULL'
Label: "Id": 1, "DataSet_id": 1, "Image_id": 1, "Label_class_id": 1, "RegionType": 1, "Region": b'{\"x\": 10.1, \"y\": 10.1, \"w\": 10.5, \"h\": 10.6, \"severe\": 7.7}'
LabelClass: "Id": 1, "Name": "裂纹", "Color": "#D1FF82", "ShortCut": "D", "Additional_data": b'{\"labelClassType\": 0, \"linearity\": 3, \"modelIndex\": 0, \"productId\": 0}'
TagSet: "Id": 1, "Name": "默认", "SpecialOp": -1, "ShortCut": "D", "Additional_data": b'NULL'
"""

class SQLAgentQwenAdvanced(SQLAgentBase):
    """
    高级版本：手动构建Agent链，使用自定义解析器
    
    注意：这种方法需要手动组装Agent的各个组件
    建议先尝试SQLAgentQwen的简单方案
    """
    
    def __init__(self, model_name: str = None, database_uri: str = None):
        super().__init__(model_name, database_uri)
    
    def create_agent(self):
        """使用自定义Qwen解析器创建Agent"""
        from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
        from langchain.agents.agent import AgentExecutor
        from langchain.agents.mrkl.base import ZeroShotAgent
        from langchain.chains import LLMChain
        
        # 1. 创建SQL工具包
        # toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)  # ✅ 使用包装后的 LLM
        from sql_agent.custom_sql_tools import CustomSQLDatabaseToolkit
        toolkit = CustomSQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        
        # 2. 创建提示词模板
        prefix = """你是一个 SQL 数据库交互代理。
    
关键规则：
1. 使用 sql_db_query_checker 检查 SQL 后，**必须**使用 sql_db_query 执行查询
2. **绝对不能**在没有实际执行查询的情况下给出答案
3. Final Answer 必须基于 sql_db_query 的实际执行结果
4. 如果问题与数据库无关，直接返回"I don't know"作为答案
5. 如果问题与数据库有关，但无法通过工具获取答案，直接返回"I don't know"作为答案
6. Observation: the result of the action 必须包含实际的查询结果，不能由你产生

使用工具:
sql_db_list_tables - 列出数据库中的所有表
sql_db_schema - 获取表的结构信息
sql_db_query_checker - 检查SQL语句是否正确
sql_db_query - 执行SQL语句

---schema---:
Dataset: CREATE TABLE DataSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Additional_data BLOB)
Image: CREATE TABLE Image(Id INTEGER PRIMARY KEY NOT NULL,Path TEXT NOT NULL,DataSet_id INTEGER KEY NOT NULL,Tag_id INTEGER KEY NOT NULL,Additional_data BLOB,UNIQUE(Path,DataSet_id)FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Tag_id) REFERENCES TagSet(Id))
Label: CREATE TABLE Label(Id INTEGER PRIMARY KEY NOT NULL,DataSet_id INTEGER NOT NULL,Image_id INTEGER NOT NULL,Label_class_id INTEGER NOT NULL,RegionType INTEGER NOT NULL,Region BLOB,FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Image_id) REFERENCES Image(Id),FOREIGN KEY(Label_class_id) REFERENCES LabelClass(Id))
TagSet: CREATE TABLE TagSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,SpecialOp INTEGER DEFAULT (-1),ShortCut TEXT,Additional_data BLOB)
LabelClass: CREATE TABLE LabelClass(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Color TEXT UNIQUE DEFAULT NULL,ShortCut TEXT,Additional_data BLOB)

---single row sample---:
{single_row_sample}

---重要信息---
Label的Region是json格式的数据，包含x,y,w,h,severe五个字段。x,y,w,h的单位是像素，代表一个矩形的位置，severe的单位是数值，代表标注严重程度。
LabelClass的Additional_data是json格式的数据，包含labelClassType,linearity,modelIndex,productId四个字段。labelClassType的单位是数值，代表标注类型。linearity的单位是数值，代表标注线性程度。modelIndex的单位是数值，代表标注模型索引。productId的单位是数值，代表标注产品ID。
LabelClass的Color是颜色代码，代表标注颜色。
Dataset的Additional_data是json格式的数据，包含locked这个字段。locked的单位是布尔值，代表数据集是否锁定。
"""
        suffix = """Begin!

IMPORTANT: Do NOT use <think> tags or any XML-style reasoning tags in your output.
Use this exact format:

Question: {input}
Thought: I should think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""
        
        # 3. 创建提示词
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "agent_scratchpad", "single_row_sample"],
        )
        
        # 4. 创建LLM链
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)  # ✅ 使用包装后的 LLM
        
        # 5. 创建Agent，使用自定义解析器
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in tools],
            output_parser=get_qwen_react_parser(),  # 使用自定义解析器！
        )
        
        # 6. 创建AgentExecutor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=None,
        )
        
        return agent_executor

    def query(self, question: str, callback: list = None):
        """执行查询"""
        if callback:
            return self.agent.invoke({"input": question, "single_row_sample": single_row_sample}, config={"callbacks":callback})
        else:
            return self.agent.invoke({"input": question, "single_row_sample": single_row_sample})

# 默认使用简单版本
SQLAgentQwen3 = SQLAgentQwenAdvanced

