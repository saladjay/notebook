"""
展示 LangChain SQL Agent 发送给模型的完整 Prompt
用于调试和学习 Agent 的工作原理
"""
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List
from config import get_config
import json


class PromptCaptureCallback(BaseCallbackHandler):
    """
    自定义回调处理器，捕获发送给LLM的完整prompt
    """
    
    def __init__(self):
        self.prompts = []
        self.current_prompt = None
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """当LLM开始时调用"""
        print("\n" + "="*80)
        print("捕获到发送给模型的 Prompt")
        print("="*80)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Prompt #{i}:")
            print("-" * 80)
            print(prompt)
            print("-" * 80)
            
            # 保存到列表
            self.prompts.append({
                'index': i,
                'prompt': prompt,
                'length': len(prompt)
            })
        
        print(f"\nPrompt 统计:")
        print(f"  - 字符数: {len(prompts[0]) if prompts else 0}")
        print(f"  - 行数: {prompts[0].count(chr(10)) + 1 if prompts else 0}")
        print("="*80 + "\n")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """当LLM结束时调用"""
        print("\n" + "="*80)
        print("模型响应")
        print("="*80)
        
        # 获取响应文本
        if hasattr(response, 'generations') and response.generations:
            for i, generation in enumerate(response.generations[0], 1):
                if hasattr(generation, 'text'):
                    print(f"\n响应 #{i}:")
                    print("-" * 80)
                    print(generation.text)
                    print("-" * 80)
        
        print("="*80 + "\n")
    
    def get_prompts(self):
        """获取捕获的所有prompts"""
        return self.prompts


def show_prompt_structure(question: str = "train数据集里有多少张图片"):
    """
    展示指定问题的完整 Prompt 结构
    
    Args:
        question: 要查询的问题
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🔍 SQL Agent Prompt 查看器                               ║
║     展示发送给模型的完整 Prompt                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 加载配置
    config = get_config()
    
    print(f"配置信息:")
    print(f"  - 模型: {config['ollama_model']}")
    print(f"  - 数据库: {config['database_uri']}")
    print(f"  - 问题: {question}")
    
    # 初始化
    from sql_agent_base import SQLAgentBase
    
    print(f"\n{'='*80}")
    print("📋 初始化 SQL Agent...")
    print(f"{'='*80}")
    
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    # 修正URI
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    
    db = SQLDatabase.from_uri(db_uri)
    
    print(f"✅ 连接成功")
    print(f"   可用表: {db.get_usable_table_names()}")
    
    # 创建回调处理器
    callback = PromptCaptureCallback()
    
    # 创建 Agent（使用回调）
    print(f"\n{'='*80}")
    print("🤖 创建 SQL Agent（带 Prompt 捕获）...")
    print(f"{'='*80}")
    
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # 显示详细过程
        handle_parsing_errors=True,
        max_iterations=10,
    )
    
    # 执行查询
    print(f"\n{'='*80}")
    print(f"❓ 执行查询: {question}")
    print(f"{'='*80}")
    
    try:
        result = agent.invoke(
            {"input": question},
            config={"callbacks": [callback]}  # 添加回调
        )
        
        print(f"\n{'='*80}")
        print("✅ 查询完成")
        print(f"{'='*80}")
        print(f"\n最终答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"\n❌ 查询失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 显示捕获的所有 prompts
    prompts = callback.get_prompts()
    
    if prompts:
        print(f"\n\n{'='*80}")
        print("📊 Prompt 汇总")
        print(f"{'='*80}")
        print(f"总共捕获了 {len(prompts)} 个 Prompt")
        
        for p in prompts:
            print(f"\nPrompt #{p['index']}:")
            print(f"  - 长度: {p['length']} 字符")
            print(f"  - 行数: {p['prompt'].count(chr(10)) + 1}")
    
    return prompts


def analyze_prompt_components():
    """
    分析 SQL Agent Prompt 的组成部分
    不实际执行查询，只展示 Prompt 模板
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     📖 SQL Agent Prompt 结构分析                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
    
    print("\n" + "="*80)
    print("📝 Prompt 组成部分")
    print("="*80)
    
    print("\n1️⃣  SQL_PREFIX (系统提示)")
    print("-" * 80)
    print(SQL_PREFIX)
    print("-" * 80)
    
    print("\n2️⃣  SQL_SUFFIX (格式要求)")
    print("-" * 80)
    print(SQL_SUFFIX)
    print("-" * 80)
    
    print("\n" + "="*80)
    print("📊 Prompt 结构说明")
    print("="*80)
    
    print("""
完整的 Prompt 包含以下部分：

1. 系统提示 (SQL_PREFIX)
   - 角色定义：你是一个SQL专家
   - 任务说明：回答关于数据库的问题
   - 注意事项：查询限制、安全规则等

2. 数据库Schema
   - 表结构信息
   - 字段名称和类型
   - 自动从数据库获取

3. 可用工具列表
   - sql_db_query: 执行SQL查询
   - sql_db_schema: 获取表结构
   - sql_db_list_tables: 列出所有表

4. 格式要求 (SQL_SUFFIX)
   - ReAct 格式说明
   - Action/Action Input 规范
   - 示例格式

5. 用户问题
   - 当前的查询问题

6. 历史对话（如果有）
   - Agent的思考过程
   - 工具调用记录
   - 观察结果
    """)
    
    print("\n" + "="*80)
    print("💡 如何查看实际的 Prompt")
    print("="*80)
    print("""
方法1: 使用本脚本（推荐）
  python show_agent_prompt.py

方法2: 在代码中添加回调
  callback = PromptCaptureCallback()
  agent.invoke({"input": question}, config={"callbacks": [callback]})

方法3: 设置 verbose=True
  create_sql_agent(..., verbose=True)
  
方法4: 使用 LangSmith (官方工具)
  https://smith.langchain.com/
    """)


def save_prompt_to_file(question: str, output_file: str = "agent_prompt_output.txt"):
    """
    保存 Prompt 到文件
    
    Args:
        question: 查询问题
        output_file: 输出文件名
    """
    print(f"\n💾 保存 Prompt 到文件: {output_file}")
    
    import sys
    from io import StringIO
    
    # 捕获输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        prompts = show_prompt_structure(question)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"✅ 已保存到: {output_file}")
    
    return output_file


def interactive_prompt_viewer():
    """
    交互式 Prompt 查看器
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     💬 交互式 Prompt 查看器                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n选择操作:")
    print("  [1] 查看指定问题的 Prompt")
    print("  [2] 分析 Prompt 结构")
    print("  [3] 保存 Prompt 到文件")
    print("  [0] 退出")
    
    while True:
        choice = input("\n请选择 (0-3): ").strip()
        
        if choice == '0':
            print("👋 再见！")
            break
        
        elif choice == '1':
            question = input("\n请输入问题（回车使用默认）: ").strip()
            if not question:
                question = "train数据集有多少张图片？"
            show_prompt_structure(question)
            input("\n按 Enter 继续...")
        
        elif choice == '2':
            analyze_prompt_components()
            input("\n按 Enter 继续...")
        
        elif choice == '3':
            question = input("\n请输入问题（回车使用默认）: ").strip()
            if not question:
                question = "train数据集有多少张图片？"
            
            filename = input("文件名（回车使用默认）: ").strip()
            if not filename:
                filename = "agent_prompt_output.txt"
            
            save_prompt_to_file(question, filename)
            input("\n按 Enter 继续...")
        
        else:
            print("❌ 无效选项")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "analyze":
            # 分析 Prompt 结构
            analyze_prompt_components()
        
        elif command == "save":
            # 保存到文件
            question = sys.argv[2] if len(sys.argv) > 2 else "train数据集有多少张图片？"
            filename = sys.argv[3] if len(sys.argv) > 3 else "agent_prompt_output.txt"
            save_prompt_to_file(question, filename)
        
        elif command == "interactive":
            # 交互模式
            interactive_prompt_viewer()
        
        else:
            # 显示指定问题的 Prompt
            question = command
            show_prompt_structure(question)
    
    else:
        # 默认：显示默认问题的 Prompt
        show_prompt_structure("train数据集有多少张图片？/nothink")
        # show_prompt_structure("how many images are there in the train dataset?")
        
        print("\n\n" + "="*80)
        print("💡 更多用法")
        print("="*80)
        print("""
# 查看默认问题
python show_agent_prompt.py

# 查看指定问题
python show_agent_prompt.py "工资最高的员工是谁？"

# 分析 Prompt 结构
python show_agent_prompt.py analyze

# 保存到文件
python show_agent_prompt.py save "问题" output.txt

# 交互模式
python show_agent_prompt.py interactive
        """)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

