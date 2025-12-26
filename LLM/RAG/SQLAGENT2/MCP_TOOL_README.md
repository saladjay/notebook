# LangChain调用自定义MCP工具示例

本目录包含使用LangChain调用自定义MCP（Model Context Protocol）工具的完整示例。

## 什么是MCP？

MCP（Model Context Protocol）是Anthropic提出的一个开放协议，用于让AI助手与外部工具和数据源进行标准化交互。通过MCP，AI可以：

- 访问外部数据源（数据库、文件系统、API等）
- 执行各种操作（查询、读写、计算等）
- 以标准化的方式与工具集成

## 文件说明

### 1. `mcp_tool_example.py` - 基础MCP工具示例

这个文件展示了如何：
- 创建自定义的MCP工具（继承`BaseTool`）
- 定义工具的输入参数schema（使用Pydantic）
- 实现工具的同步和异步执行逻辑
- 将工具集成到LangChain Agent中
- 直接调用工具（不使用Agent）

**包含的工具示例：**
- `DatabaseQueryTool`: 数据库查询工具
- `FileReadTool`: 文件读取工具
- `WebSearchTool`: 网络搜索工具

### 2. `mcp_server_example.py` - MCP服务器连接示例

这个文件展示了如何：
- 连接到真实的MCP服务器
- 将MCP服务器提供的工具包装为LangChain工具
- 动态获取和使用MCP服务器工具

## 安装依赖

```bash
# 基础依赖（已在requirements.txt中）
pip install langchain langchain-openai pydantic

# 如果需要连接真实MCP服务器
pip install mcp
```

## 使用方法

### 方式1: 使用自定义MCP工具

```python
from mcp_tool_example import DatabaseQueryTool, FileReadTool, create_mcp_agent
from langchain_openai import ChatOpenAI

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建工具列表
tools = [
    DatabaseQueryTool(),
    FileReadTool(),
]

# 创建Agent
agent = create_mcp_agent(llm, tools)

# 使用Agent
result = agent.invoke({
    "input": "查询用户表中的前5条记录"
})
print(result['output'])
```

### 方式2: 直接调用工具

```python
from mcp_tool_example import DatabaseQueryTool

# 创建工具实例
tool = DatabaseQueryTool()

# 直接调用
result = tool.run({
    "query": "SELECT * FROM users LIMIT 5",
    "database": "mydb"
})
print(result)
```

### 方式3: 连接MCP服务器

```python
from mcp_server_example import MockMCPClient, create_mcp_tools_from_server
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

# 连接到MCP服务器
mcp_client = MockMCPClient(server_url="http://localhost:8000")

# 获取工具
tools = create_mcp_tools_from_server(mcp_client)

# 创建Agent并使用
llm = ChatOpenAI(model="gpt-3.5-turbo")
# ... 创建agent的代码
```

## 创建自定义MCP工具的步骤

### 1. 定义输入参数Schema

```python
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param1: str = Field(description="参数1的描述")
    param2: int = Field(default=10, description="参数2的描述")
```

### 2. 创建工具类

```python
from langchain.tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "工具的描述信息"
    args_schema: type[BaseModel] = MyToolInput
    
    def _run(self, param1: str, param2: int = 10) -> str:
        """同步执行逻辑"""
        # 实现工具的具体逻辑
        return "工具执行结果"
    
    async def _arun(self, param1: str, param2: int = 10) -> str:
        """异步执行逻辑"""
        return self._run(param1, param2)
```

### 3. 使用工具

```python
# 方式1: 在Agent中使用
tools = [MyCustomTool()]
agent = create_mcp_agent(llm, tools)

# 方式2: 直接调用
tool = MyCustomTool()
result = tool.run({"param1": "value1", "param2": 20})
```

## 工具设计最佳实践

1. **清晰的描述**: 工具的描述应该清楚地说明工具的用途、输入参数和返回值
2. **参数验证**: 使用Pydantic模型进行参数验证，确保输入正确
3. **错误处理**: 在工具实现中添加适当的错误处理
4. **返回值格式**: 返回结构化的数据（如JSON），便于Agent理解和使用
5. **异步支持**: 如果工具可能执行耗时操作，实现异步版本

## 与真实MCP服务器集成

要连接真实的MCP服务器，你需要：

1. **安装MCP Python SDK**:
   ```bash
   pip install mcp
   ```

2. **配置MCP服务器连接**:
   ```python
   from mcp import ClientSession, StdioServerParameters
   from mcp.client.stdio import stdio_client
   
   server_params = StdioServerParameters(
       command="python",
       args=["-m", "your_mcp_server"],
   )
   ```

3. **使用MCP客户端**:
   参考`mcp_server_example.py`中的`real_mcp_example()`函数

更多信息请参考：
- [MCP官方文档](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## 运行示例

```bash
# 运行基础示例
python mcp_tool_example.py

# 运行服务器连接示例
python mcp_server_example.py
```

## 注意事项

1. **API密钥**: 使用OpenAI API时需要设置`OPENAI_API_KEY`环境变量
2. **本地LLM**: 如果使用本地LLM服务，需要修改`base_url`参数
3. **工具权限**: 确保工具有足够的权限执行所需操作（如文件读写、数据库访问等）
4. **安全性**: 在生产环境中，应该对工具调用进行权限验证和输入验证

## 扩展阅读

- [LangChain工具文档](https://python.langchain.com/docs/modules/tools/)
- [LangChain Agent文档](https://python.langchain.com/docs/modules/agents/)
- [Pydantic文档](https://docs.pydantic.dev/)



