"""
意图识别与路由模块
判断用户问题的类型并路由到相应的处理器
"""
from typing import Literal
# from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

Dataset_schema = "CREATE TABLE DataSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Additional_data BLOB)" 
Image_schema = "CREATE TABLE Image(Id INTEGER PRIMARY KEY NOT NULL,Path TEXT NOT NULL,DataSet_id INTEGER KEY NOT NULL,Tag_id INTEGER KEY NOT NULL,Additional_data BLOB,UNIQUE(Path,DataSet_id)FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Tag_id) REFERENCES TagSet(Id))"
Label_schema = "CREATE TABLE Label(Id INTEGER PRIMARY KEY NOT NULL,DataSet_id INTEGER NOT NULL,Image_id INTEGER NOT NULL,Label_class_id INTEGER NOT NULL,RegionType INTEGER NOT NULL,Region BLOB,FOREIGN KEY(DataSet_id) REFERENCES DataSet(Id),FOREIGN KEY(Image_id) REFERENCES Image(Id),FOREIGN KEY(Label_class_id) REFERENCES LabelClass(Id))"
TagSet_schema = "CREATE TABLE TagSet(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,SpecialOp INTEGER DEFAULT (-1),ShortCut TEXT,Additional_data BLOB)"
LabelClass_schema = "CREATE TABLE LabelClass(Id INTEGER PRIMARY KEY NOT NULL,Name TEXT NOT NULL UNIQUE,Color TEXT UNIQUE DEFAULT NULL,ShortCut TEXT,Additional_data BLOB)"

schema = f"""
{Dataset_schema}
{Image_schema}
{Label_schema}
{TagSet_schema}
{LabelClass_schema}
"""

class IntentClassification(BaseModel):
    """意图分类结果"""
    intent: Literal["simple_query", "complex_query", "clarification_needed"] = Field(
        description="问题类型：simple_query(简单查询)、complex_query(复杂多步查询)、clarification_needed(需要澄清)"
    )
    reasoning: str = Field(description="分类理由")
    confidence: float = Field(description="置信度(0-1)", ge=0.0, le=1.0)


class IntentRouter:
    """意图识别与路由器"""
    
    def __init__(self, llm):
        """
        初始化意图路由器
        
        Args:
            llm: 语言模型
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=IntentClassification)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL Agent的意图识别模块。你需要分析用户的自然语言问题，判断其类型。

问题类型定义：
1. simple_query（简单查询）：
   - 只需要查询单个表或简单的JOIN
   - 不需要多步骤推理
   - 问题明确，不需要澄清
   - 例如："查询符合条件的数据集"、"统计某个标注的图片数量"

2. complex_query（复杂查询）：
   - 需要需要分解为多个子任务
   - 需要多步骤推理或计算, 通常需要解析计算region的json中的坐标信息
   - 可能需要中间结果
   - 例如："标注区域位于图片左上角的图片数量"、"标注重合度（iou)较高的图片有哪些"

3. clarification_needed（需要澄清）：
   - 问题模糊或有歧义
   - 缺少关键信息
   - 需要用户提供更多细节
   - 例如："查询某个数据集的图片数量"（哪个数据集？）

请分析用户问题并返回分类结果。

{format_instructions}
"""),
            ("user", "用户问题: {question}\n\n可用的数据库表: {schema}")
        ])
    
    def classify_intent(
        self, 
        question: str, 
        tables: list,
        context: str = ""
    ) -> IntentClassification:
        """
        分类用户意图
        
        Args:
            question: 用户问题
            tables: 可用的数据库表列表
            context: 额外的上下文信息
            
        Returns:
            IntentClassification: 分类结果
        """
        chain = self.prompt | self.llm | self.parser
        # schema = 
        result = chain.invoke({
            "question": question,
            # "tables": ", ".join(tables),
            "schema": schema,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def should_clarify(self, classification: IntentClassification) -> bool:
        """判断是否需要澄清"""
        return classification.intent == "clarification_needed"
    
    def is_simple_query(self, classification: IntentClassification) -> bool:
        """判断是否为简单查询"""
        return classification.intent == "simple_query"
    
    def is_complex_query(self, classification: IntentClassification) -> bool:
        """判断是否为复杂查询"""
        return classification.intent == "complex_query"

def _get_bailian_llm():
    try:
        from langchain_qwq import ChatQwen
    except ImportError:
        print("请先安装 langchain_qwq: pip install langchain-qwq")
        return None
    chatLLM = ChatQwen(
        api_key=BailianAPIKey,
        base_url=BailianURI,
        model=BailianModel,  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        # other params...
        temperature=0.1,
        top_p=0.95,
        enable_thinking=False,
        # thinking_budget=1000,
    )
    return chatLLM

BailianURI = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BailianModel = "qwen3-max-2025-09-23"
BailianAPIKey = "sk-098789f6e2be43bd8bf2befc2ee24331"

def _get_vllm_llm():
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("请先安装 langchain_openai: pip install langchain-openai")
        return None
    
    # 使用OpenAI接口调用vLLM服务
    chatLLM = ChatOpenAI(
        api_key=VLLMAPIKey,  # vLLM通常不需要API Key，设置为"EMPTY"或任意字符串
        base_url=VLLMURI,    # vLLM服务的地址
        model=VLLMModel,     # 模型名称
        temperature=0.1,
        max_tokens=2048,
        top_p=0.9,
        timeout=60,          # 设置超时时间
    )
    return chatLLM

# vLLM 配置 - 局域网内 vLLM 服务
VLLMURI = "http://192.168.2.197:8000/v1"  # vLLM OpenAI兼容接口地址
VLLMModel = "dihugesql"  # 修改为你的模型名称
VLLMAPIKey = "EMPTY"  # vLLM 通常不需要 API Key，设置为 "EMPTY" 或任意字符串

def _get_ollama_llm():
    try:
        from langchain_ollama import OllamaLLM  # 新的、推荐的方式
    except ImportError:
        print("请先安装 langchain_community: pip install -U langchain-ollama")
        return
    llm = OllamaLLM(
        model=OllamaModel,
        base_url=OllamaURI,
        model_kwargs={
            "mirostat": 2,          # 启用 Mirostat 2.0
            "mirostat_tau": 5.0,    # 目标困惑度（推荐 3.0-5.0）
            "mirostat_eta": 0.1,    # 学习率
            "temperature": 0.7,
            "num_predict": 2048     # 增加最大输出长度（默认128太小，防止输出被截断）
        },
        # 添加系统提示，禁止使用<think>标签
        # system="You are a helpful AI assistant. When answering questions, do NOT use <think> tags or any XML-style tags for internal reasoning. Simply provide your thoughts and actions in plain text using the specified format."
    )
    return llm

OllamaURI = "http://192.168.2.193:11434"
OllamaModel = "qwen3:1.7b"
if __name__ == "__main__":
    llm = _get_ollama_llm()
    router = IntentRouter(llm)
    result = router.classify_intent("哪些图像的标注区域之间存在大量重叠， 两个标注框坐标相差10个像素，可能表示重复标注？", schema)
    print(result)
    result = router.classify_intent("哪些Tag与小于80x90标注同时出现？", schema)
    print(result)
    result = router.classify_intent("哪些图像包含数量异常多的标注，高于同一个数据集平均值的两倍，可能暗示标注冗余或误标？", schema)
    print(result)