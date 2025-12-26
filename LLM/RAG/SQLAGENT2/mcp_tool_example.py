"""
使用LangChain调用自定义MCP工具的示例

MCP (Model Context Protocol) 是Anthropic提出的协议，用于让AI助手与外部工具和数据源交互。
本示例展示如何创建自定义MCP工具并通过LangChain调用。
"""

from typing import Optional, List, Dict, Any
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json


# ==================== 自定义MCP工具定义 ====================

class DatabaseQueryInput(BaseModel):
    """数据库查询工具的输入参数"""
    query: str = Field(description="要执行的SQL查询语句")
    database: Optional[str] = Field(default="default", description="数据库名称")


class DatabaseQueryTool(BaseTool):
    """自定义MCP工具：数据库查询工具"""
    
    name: str = "database_query"
    description: str = """
    执行SQL查询并返回结果。用于查询数据库中的信息。
    
    输入参数：
    - query: SQL查询语句（必需）
    - database: 数据库名称（可选，默认为'default'）
    
    返回：查询结果的JSON格式字符串
    """
    args_schema: type[BaseModel] = DatabaseQueryInput
    
    def _run(self, query: str, database: str = "default") -> str:
        """执行工具逻辑"""
        # 这里模拟数据库查询，实际使用时应该连接真实数据库
        print(f"[MCP工具] 执行查询: {query} (数据库: {database})")
        
        # 模拟查询结果
        mock_results = {
            "SELECT * FROM users LIMIT 5": [
                {"id": 1, "name": "张三", "age": 25},
                {"id": 2, "name": "李四", "age": 30},
                {"id": 3, "name": "王五", "age": 28},
            ],
            "SELECT COUNT(*) as count FROM users": [{"count": 100}],
        }
        
        result = mock_results.get(query, [{"message": "查询执行成功", "rows": 0}])
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    async def _arun(self, query: str, database: str = "default") -> str:
        """异步执行工具逻辑"""
        return self._run(query, database)


class FileReadInput(BaseModel):
    """文件读取工具的输入参数"""
    file_path: str = Field(description="要读取的文件路径")
    encoding: Optional[str] = Field(default="utf-8", description="文件编码")


class FileReadTool(BaseTool):
    """自定义MCP工具：文件读取工具"""
    
    name: str = "read_file"
    description: str = """
    读取文件内容。用于读取本地文件系统中的文件。
    
    输入参数：
    - file_path: 文件路径（必需）
    - encoding: 文件编码（可选，默认为'utf-8'）
    
    返回：文件内容字符串
    """
    args_schema: type[BaseModel] = FileReadInput
    
    def _run(self, file_path: str, encoding: str = "utf-8") -> str:
        """执行工具逻辑"""
        print(f"[MCP工具] 读取文件: {file_path} (编码: {encoding})")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return f"文件内容（前500字符）:\n{content[:500]}"
        except FileNotFoundError:
            return f"错误: 文件 '{file_path}' 不存在"
        except Exception as e:
            return f"错误: 读取文件时发生异常 - {str(e)}"
    
    async def _arun(self, file_path: str, encoding: str = "utf-8") -> str:
        """异步执行工具逻辑"""
        return self._run(file_path, encoding)


class WebSearchInput(BaseModel):
    """网络搜索工具的输入参数"""
    query: str = Field(description="搜索关键词")
    max_results: Optional[int] = Field(default=5, description="最大返回结果数")


class WebSearchTool(BaseTool):
    """自定义MCP工具：网络搜索工具"""
    
    name: str = "web_search"
    description: str = """
    在网络上搜索信息。用于获取最新的网络信息。
    
    输入参数：
    - query: 搜索关键词（必需）
    - max_results: 最大返回结果数（可选，默认为5）
    
    返回：搜索结果列表的JSON格式字符串
    """
    args_schema: type[BaseModel] = WebSearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """执行工具逻辑"""
        print(f"[MCP工具] 搜索: {query} (最大结果数: {max_results})")
        
        # 这里模拟网络搜索，实际使用时应该调用真实的搜索API
        mock_results = [
            {
                "title": f"关于'{query}'的搜索结果1",
                "url": f"https://example.com/result1",
                "snippet": f"这是关于'{query}'的相关信息..."
            },
            {
                "title": f"关于'{query}'的搜索结果2",
                "url": f"https://example.com/result2",
                "snippet": f"更多关于'{query}'的详细信息..."
            }
        ]
        
        return json.dumps(mock_results[:max_results], ensure_ascii=False, indent=2)
    
    async def _arun(self, query: str, max_results: int = 5) -> str:
        """异步执行工具逻辑"""
        return self._run(query, max_results)


# ==================== 使用LangChain Agent调用MCP工具 ====================

def create_mcp_agent(llm, tools: List[BaseTool]):
    """
    创建使用MCP工具的LangChain Agent
    
    Args:
        llm: 语言模型实例
        tools: MCP工具列表
        
    Returns:
        AgentExecutor: 配置好的Agent执行器
    """
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，可以使用以下工具来帮助用户：
        
可用工具：
{tools}

使用工具时，请遵循以下规则：
1. 仔细分析用户的问题，确定需要使用哪些工具
2. 调用工具时，确保参数格式正确
3. 根据工具返回的结果，给出清晰的回答
4. 如果工具调用失败，尝试其他方法或向用户说明情况

请用中文回答用户的问题。"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 显示详细执行过程
        handle_parsing_errors=True,  # 处理解析错误
        max_iterations=10,  # 最大迭代次数
    )
    
    return agent_executor


def main():
    """主函数：演示如何使用MCP工具"""
    
    # 初始化LLM（这里使用OpenAI兼容接口，你可以替换为其他LLM）
    # 注意：需要设置OPENAI_API_KEY环境变量或直接传入api_key参数
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        # api_key="your-api-key",  # 如果需要，可以在这里设置
        # base_url="http://localhost:8000/v1",  # 如果使用本地LLM服务
    )
    
    # 创建MCP工具列表
    mcp_tools = [
        DatabaseQueryTool(),
        FileReadTool(),
        WebSearchTool(),
    ]
    
    # 创建Agent
    agent = create_mcp_agent(llm, mcp_tools)
    
    # 示例1: 使用数据库查询工具
    print("=" * 60)
    print("示例1: 数据库查询")
    print("=" * 60)
    result1 = agent.invoke({
        "input": "查询用户表中的前5条记录"
    })
    print(f"\n结果: {result1['output']}\n")
    
    # 示例2: 使用文件读取工具
    print("=" * 60)
    print("示例2: 文件读取")
    print("=" * 60)
    result2 = agent.invoke({
        "input": "读取当前目录下的requirements.txt文件"
    })
    print(f"\n结果: {result2['output']}\n")
    
    # 示例3: 使用网络搜索工具
    print("=" * 60)
    print("示例3: 网络搜索")
    print("=" * 60)
    result3 = agent.invoke({
        "input": "搜索关于LangChain的最新信息"
    })
    print(f"\n结果: {result3['output']}\n")
    
    # 示例4: 组合使用多个工具
    print("=" * 60)
    print("示例4: 组合使用工具")
    print("=" * 60)
    result4 = agent.invoke({
        "input": "先查询用户总数，然后搜索关于用户数据分析的最佳实践"
    })
    print(f"\n结果: {result4['output']}\n")


# ==================== 直接调用MCP工具（不使用Agent） ====================

def direct_tool_usage_example():
    """演示如何直接调用MCP工具（不使用Agent）"""
    print("=" * 60)
    print("直接调用MCP工具示例")
    print("=" * 60)
    
    # 创建工具实例
    db_tool = DatabaseQueryTool()
    file_tool = FileReadTool()
    search_tool = WebSearchTool()
    
    # 直接调用工具
    print("\n1. 直接调用数据库查询工具:")
    result1 = db_tool.run({"query": "SELECT COUNT(*) as count FROM users"})
    print(result1)
    
    print("\n2. 直接调用文件读取工具:")
    result2 = file_tool.run({"file_path": "requirements.txt"})
    print(result2)
    
    print("\n3. 直接调用网络搜索工具:")
    result3 = search_tool.run({"query": "Python编程", "max_results": 3})
    print(result3)


if __name__ == "__main__":
    print("LangChain MCP工具调用示例\n")
    
    # 方式1: 使用Agent自动调用工具
    try:
        main()
    except Exception as e:
        print(f"\n使用Agent时发生错误: {e}")
        print("提示: 请确保已设置OPENAI_API_KEY环境变量，或修改代码使用其他LLM")
    
    # 方式2: 直接调用工具
    print("\n" + "=" * 60)
    direct_tool_usage_example()

