"""
增强版 SQL Agent - 处理小模型的输出解析问题
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from sql_agent_base import SQLAgentBase
import re


class SQLAgentDemoEnhanced(SQLAgentBase):
    """
    增强版 SQL Agent
    - 自定义错误处理
    - 处理小模型输出格式问题
    """
    
    def __init__(self, model_name: str = None, database_uri: str = None):
        super().__init__(model_name, database_uri)
    
    def create_agent(self):
        """创建带有增强错误处理的Agent"""
        return create_sql_agent(
            llm=self.llm,  # ✅ 现在 LLMWrapper 继承自 BaseLanguageModel，可以直接使用
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=self._handle_parsing_error,  # 自定义错误处理
            max_iterations=10,
            max_execution_time=None,
        )
    
    @staticmethod
    def _handle_parsing_error(error) -> str:
        """
        自定义解析错误处理
        当模型输出格式不正确时，返回提示信息
        """
        error_str = str(error)
        
        # 检查是否包含 <think> 标签
        if '<think>' in error_str or '</think>' in error_str:
            return (
                "你的输出包含了 <think> 标签，这不是正确的格式。"
                "请直接输出 Action 和 Action Input，不要添加思考标签。\n"
                "正确格式示例：\n"
                "Action: sql_db_list_tables\n"
                "Action Input: \"\""
            )
        
        # 检查是否缺少 Action Input
        if 'Action Input' not in error_str and 'Action:' in error_str:
            return (
                "你的输出缺少 Action Input 字段。"
                "请按照以下格式输出：\n"
                "Action: [工具名称]\n"
                "Action Input: [输入参数]"
            )
        
        # 通用错误提示
        return (
            f"输出格式错误。请严格按照以下格式：\n"
            f"Action: [工具名称]\n"
            f"Action Input: [输入参数]\n\n"
            f"可用工具：sql_db_list_tables, sql_db_schema, sql_db_query\n"
            f"错误详情: {str(error)[:200]}"
        )
    
    def query_with_retry(self, question: str, max_retries: int = 2):
        """
        带重试机制的查询
        如果遇到解析错误，会尝试简化问题
        """
        for attempt in range(max_retries + 1):
            try:
                print(f"\n{'='*60}")
                if attempt > 0:
                    print(f"🔄 重试第 {attempt} 次...")
                print(f"❓ 问题: {question}")
                print(f"{'='*60}")
                
                result = self.agent.invoke({"input": question})
                return result
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ 错误 (尝试 {attempt + 1}/{max_retries + 1}): {error_msg[:200]}")
                
                if attempt < max_retries:
                    # 简化问题或给出提示
                    if "parse" in error_msg.lower():
                        print("⚠️  模型输出格式不正确，尝试简化问题...")
                        # 可以在这里简化问题
                    else:
                        print("⚠️  遇到错误，正在重试...")
                else:
                    print(f"\n💡 建议：")
                    print(f"  1. 使用更大的模型（如 qwen2.5:7b）")
                    print(f"  2. 简化你的问题")
                    print(f"  3. 直接使用SQL查询而不是自然语言")
                    raise


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     🚀 增强版 SQL Agent Demo                             ║
║     处理小模型输出解析问题                                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 创建Agent
        agent = SQLAgentDemoEnhanced()
        
        print("\n" + "="*60)
        print("📝 开始测试查询...")
        print("="*60)
        
        # 简单查询
        test_queries = [
            "列出所有表名",
            "employees表有多少条记录？",
        ]
        
        for query in test_queries:
            try:
                result = agent.query_with_retry(query)
                print(f"\n✅ 成功!")
                print(f"结果: {result.get('output', result)}")
            except Exception as e:
                print(f"\n❌ 最终失败: {str(e)[:300]}")
                print(f"\n⚠️  当前模型可能不适合SQL Agent任务")
                break
        
        print("\n" + "="*60)
        print("💡 模型选择建议")
        print("="*60)
        print("""
SQL Agent 推荐模型：
  ✅ qwen2.5:7b    - 最推荐（中文+SQL都很好）
  ✅ qwen2.5:3b    - 较小但可用
  ✅ llama3.2:3b   - 英文较好
  ⚠️  qwen3:0.6b   - 太小，不推荐（当前使用）

安装推荐模型：
  ollama pull qwen2.5:7b
        """)
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

