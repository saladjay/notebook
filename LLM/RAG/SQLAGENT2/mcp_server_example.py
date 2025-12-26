"""
连接真实MCP服务器的示例

这个示例展示如何通过MCP协议连接到真实的MCP服务器并使用其提供的工具。
需要安装 mcp 包: pip install mcp
"""

from typing import Optional, List, Dict, Any
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
import asyncio


# ==================== MCP客户端包装器 ====================

class MCPToolWrapper(BaseTool):
    """
    将MCP服务器提供的工具包装为LangChain工具
    
    这个类将MCP协议的工具转换为LangChain可以使用的工具格式
    """
    
    def __init__(
        self,
        mcp_tool_name: str,
        mcp_tool_description: str,
        mcp_tool_schema: Dict[str, Any],
        mcp_client,  # MCP客户端实例
        **kwargs
    ):
        """
        初始化MCP工具包装器
        
        Args:
            mcp_tool_name: MCP工具名称
            mcp_tool_description: MCP工具描述
            mcp_tool_schema: MCP工具的输入参数schema
            mcp_client: MCP客户端实例，用于调用工具
        """
        # 从MCP schema创建Pydantic模型
        properties = mcp_tool_schema.get("properties", {})
        required = mcp_tool_schema.get("required", [])
        
        # 动态创建输入模型
        fields = {}
        for prop_name, prop_info in properties.items():
            field_type = str  # 默认类型
            field_default = ... if prop_name in required else None
            field_description = prop_info.get("description", "")
            
            fields[prop_name] = (
                field_type,
                Field(
                    default=field_default,
                    description=field_description
                )
            )
        
        InputModel = type(
            f"{mcp_tool_name}Input",
            (BaseModel,),
            fields
        )
        
        super().__init__(
            name=mcp_tool_name,
            description=mcp_tool_description,
            args_schema=InputModel,
            **kwargs
        )
        
        self.mcp_client = mcp_client
        self.mcp_tool_name = mcp_tool_name
    
    def _run(self, **kwargs) -> str:
        """同步执行MCP工具"""
        try:
            # 调用MCP客户端的方法
            # 注意：实际实现取决于你使用的MCP客户端库
            result = self.mcp_client.call_tool(
                tool_name=self.mcp_tool_name,
                arguments=kwargs
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"错误: 调用MCP工具时发生异常 - {str(e)}"
    
    async def _arun(self, **kwargs) -> str:
        """异步执行MCP工具"""
        try:
            # 异步调用MCP客户端
            result = await self.mcp_client.call_tool_async(
                tool_name=self.mcp_tool_name,
                arguments=kwargs
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"错误: 调用MCP工具时发生异常 - {str(e)}"


# ==================== 模拟MCP客户端 ====================

class MockMCPClient:
    """
    模拟MCP客户端
    
    在实际使用中，你应该使用真实的MCP客户端库，例如：
    - mcp (Anthropic官方MCP Python SDK)
    - 或其他兼容MCP协议的客户端
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.tools = {}  # 存储可用的工具
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用的工具"""
        return [
            {
                "name": "database_query",
                "description": "执行SQL查询",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL查询语句"
                        },
                        "database": {
                            "type": "string",
                            "description": "数据库名称",
                            "default": "default"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "file_operations",
                "description": "文件操作工具",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "操作类型: read, write, list",
                            "enum": ["read", "write", "list"]
                        },
                        "path": {
                            "type": "string",
                            "description": "文件路径"
                        },
                        "content": {
                            "type": "string",
                            "description": "文件内容（仅write操作需要）"
                        }
                    },
                    "required": ["operation", "path"]
                }
            }
        ]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """同步调用工具"""
        print(f"[MCP客户端] 调用工具: {tool_name}, 参数: {arguments}")
        
        # 模拟工具执行
        if tool_name == "database_query":
            return {
                "success": True,
                "data": [{"id": 1, "name": "测试数据"}],
                "rows": 1
            }
        elif tool_name == "file_operations":
            operation = arguments.get("operation")
            path = arguments.get("path")
            if operation == "read":
                return {
                    "success": True,
                    "content": f"文件 {path} 的内容（模拟）"
                }
            elif operation == "list":
                return {
                    "success": True,
                    "files": ["file1.txt", "file2.txt"]
                }
        
        return {"success": False, "error": f"未知工具: {tool_name}"}
    
    async def call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """异步调用工具"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        return self.call_tool(tool_name, arguments)


# ==================== 使用示例 ====================

def create_mcp_tools_from_server(mcp_client: MockMCPClient) -> List[BaseTool]:
    """
    从MCP服务器获取工具列表并转换为LangChain工具
    
    Args:
        mcp_client: MCP客户端实例
        
    Returns:
        LangChain工具列表
    """
    tools = []
    mcp_tools = mcp_client.list_tools()
    
    for mcp_tool in mcp_tools:
        wrapper = MCPToolWrapper(
            mcp_tool_name=mcp_tool["name"],
            mcp_tool_description=mcp_tool["description"],
            mcp_tool_schema=mcp_tool["inputSchema"],
            mcp_client=mcp_client
        )
        tools.append(wrapper)
    
    return tools


def main():
    """主函数：演示如何连接MCP服务器并使用其工具"""
    
    # 1. 创建MCP客户端（连接到MCP服务器）
    print("正在连接到MCP服务器...")
    mcp_client = MockMCPClient(server_url="http://localhost:8000")
    
    # 2. 获取MCP服务器提供的工具并转换为LangChain工具
    print("获取MCP工具列表...")
    langchain_tools = create_mcp_tools_from_server(mcp_client)
    
    print(f"找到 {len(langchain_tools)} 个工具:")
    for tool in langchain_tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # 3. 初始化LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    
    # 4. 创建Agent提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，可以使用MCP服务器提供的工具来帮助用户。

可用工具：
{tools}

请根据用户的问题，选择合适的工具并正确调用。"""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 5. 创建Agent
    agent = create_openai_tools_agent(llm, langchain_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=langchain_tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    # 6. 使用Agent
    print("\n" + "=" * 60)
    print("使用MCP工具执行任务")
    print("=" * 60)
    
    result = agent_executor.invoke({
        "input": "查询数据库中的用户信息"
    })
    
    print(f"\n最终结果: {result['output']}")


# ==================== 真实MCP客户端使用示例 ====================

def real_mcp_example():
    """
    使用真实MCP客户端的示例代码（需要安装mcp包）
    
    安装: pip install mcp
    """
    try:
        # 注意：这是示例代码，实际API可能不同
        # from mcp import ClientSession, StdioServerParameters
        # from mcp.client.stdio import stdio_client
        # 
        # # 配置MCP服务器连接
        # server_params = StdioServerParameters(
        #     command="python",
        #     args=["-m", "mcp_server"],
        # )
        # 
        # # 创建客户端会话
        # async with stdio_client(server_params) as (read, write):
        #     async with ClientSession(read, write) as session:
        #         # 初始化会话
        #         await session.initialize()
        #         
        #         # 列出可用工具
        #         tools = await session.list_tools()
        #         print(f"可用工具: {tools}")
        #         
        #         # 调用工具
        #         result = await session.call_tool(
        #             "database_query",
        #             {"query": "SELECT * FROM users"}
        #         )
        #         print(f"结果: {result}")
        
        print("真实MCP客户端示例需要安装mcp包并配置MCP服务器")
        print("请参考: https://github.com/modelcontextprotocol/python-sdk")
        
    except ImportError:
        print("请先安装MCP Python SDK: pip install mcp")


if __name__ == "__main__":
    print("MCP服务器连接示例\n")
    
    try:
        main()
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("提示: 请确保已设置OPENAI_API_KEY环境变量")
    
    print("\n" + "=" * 60)
    real_mcp_example()

