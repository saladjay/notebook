"""
简单的LangChain MCP工具调用示例

这是一个最简化的示例，展示如何创建和使用自定义MCP工具。
"""

from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json


# ==================== 步骤1: 定义工具输入参数 ====================

class CalculatorInput(BaseModel):
    """计算器工具的输入参数"""
    expression: str = Field(description="要计算的数学表达式，例如: '2 + 2', '10 * 5'")


# ==================== 步骤2: 创建自定义MCP工具 ====================

class CalculatorTool(BaseTool):
    """
    自定义MCP工具：计算器
    
    这个工具可以执行基本的数学计算
    """
    name: str = "calculator"
    description: str = """
    执行数学计算。输入一个数学表达式，返回计算结果。
    
    示例:
    - 输入: "2 + 2" -> 输出: "4"
    - 输入: "10 * 5" -> 输出: "50"
    - 输入: "(3 + 4) * 2" -> 输出: "14"
    """
    args_schema: type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """执行计算"""
        try:
            # 安全地执行数学表达式
            # 注意：在生产环境中应该使用更安全的表达式解析器
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """异步执行（这里直接调用同步版本）"""
        return self._run(expression)


# ==================== 步骤3: 使用工具 ====================

def example_direct_usage():
    """示例1: 直接调用工具"""
    print("=" * 60)
    print("示例1: 直接调用工具")
    print("=" * 60)
    
    # 创建工具实例
    calculator = CalculatorTool()
    
    # 直接调用
    result1 = calculator.run({"expression": "2 + 2"})
    print(f"2 + 2 = {result1}")
    
    result2 = calculator.run({"expression": "10 * 5"})
    print(f"10 * 5 = {result2}")
    
    result3 = calculator.run({"expression": "(3 + 4) * 2"})
    print(f"(3 + 4) * 2 = {result3}")


def example_with_agent():
    """示例2: 在Agent中使用工具"""
    print("\n" + "=" * 60)
    print("示例2: 在Agent中使用工具")
    print("=" * 60)
    
    try:
        # 初始化LLM
        # 注意：需要设置OPENAI_API_KEY环境变量
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )
        
        # 创建工具列表
        tools = [CalculatorTool()]
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能助手，可以使用计算器工具来帮助用户进行数学计算。

可用工具：
{tools}

当用户需要计算时，使用calculator工具。"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建Agent
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # 显示详细执行过程
            handle_parsing_errors=True,
        )
        
        # 使用Agent
        print("\n问题: 计算 25 * 4 + 10")
        result = agent_executor.invoke({
            "input": "计算 25 * 4 + 10"
        })
        print(f"\n答案: {result['output']}")
        
        print("\n问题: 帮我算一下 (100 - 20) / 4 等于多少？")
        result = agent_executor.invoke({
            "input": "帮我算一下 (100 - 20) / 4 等于多少？"
        })
        print(f"\n答案: {result['output']}")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 请确保已设置OPENAI_API_KEY环境变量")
        print("或者修改代码使用其他LLM（如本地LLM服务）")


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("LangChain MCP工具简单示例\n")
    
    # 示例1: 直接调用（不需要LLM）
    example_direct_usage()
    
    # 示例2: 使用Agent（需要LLM）
    example_with_agent()



