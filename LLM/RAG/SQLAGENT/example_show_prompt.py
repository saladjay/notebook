"""
最简单的示例：如何在你的代码中捕获 Prompt
复制这个代码到你的项目中即可使用
"""
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List


# ============================================
# 步骤1: 创建回调类（复制这部分）
# ============================================

class PromptPrinter(BaseCallbackHandler):
    """
    简单的 Prompt 打印器
    在 LLM 调用前打印完整的 Prompt
    """
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """当 LLM 开始执行时被调用"""
        print("\n" + "="*80)
        print("📤 发送给模型的 Prompt:")
        print("="*80)
        
        for i, prompt in enumerate(prompts, 1):
            if len(prompts) > 1:
                print(f"\n--- Prompt {i}/{len(prompts)} ---")
            
            print(prompt)
            
            # 显示统计
            print(f"\n📊 统计: {len(prompt)} 字符, {prompt.count(chr(10)) + 1} 行")
        
        print("="*80 + "\n")


# ============================================
# 步骤2: 在你的代码中使用（复制这部分）
# ============================================

def example_usage():
    """示例：如何使用 PromptPrinter"""
    
    from sql_agent_base import SQLAgentBase
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     📝 Prompt 捕获示例                                       ║
║     最简单的方法：在 invoke 时添加 callback                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 1. 创建你的 Agent（正常流程）
    print("1️⃣  创建 SQL Agent...")
    agent = SQLAgentBase()
    
    # 2. 创建 Prompt 打印器
    print("2️⃣  创建 Prompt 捕获器...")
    prompt_printer = PromptPrinter()
    
    # 3. 执行查询，添加 callback 参数
    print("3️⃣  执行查询（会自动打印 Prompt）...\n")
    
    question = "train数据集里有多少张图片"
    
    try:
        result = agent.agent.invoke(
            {"input": question},
            config={"callbacks": [prompt_printer]}  # ← 关键：添加这一行
        )
        
        print("\n" + "="*80)
        print("✅ 查询完成")
        print("="*80)
        print(f"答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")


# ============================================
# 步骤3: 在任何 LangChain 组件中使用
# ============================================

def example_with_any_agent():
    """
    这个方法适用于任何 LangChain 的 Agent 或 Chain
    """
    from langchain_community.llms import Ollama
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent
    from langchain.agents.agent_types import AgentType
    from config import get_config
    
    config = get_config()
    
    # 创建 LLM
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    # 创建数据库连接
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    # 创建 Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,  # 可以关闭 verbose，我们有自己的打印器
        handle_parsing_errors=True,
    )
    
    # 使用 callback
    callback = PromptPrinter()
    
    result = agent.invoke(
        {"input": "列出所有表名"},
        config={"callbacks": [callback]}
    )
    
    print(f"\n结果: {result}")


# ============================================
# 更高级的用法：保存 Prompt 到变量
# ============================================

class PromptCapture(BaseCallbackHandler):
    """
    捕获 Prompt 到变量，方便后续处理
    """
    
    def __init__(self):
        self.prompts = []
        self.responses = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """捕获 prompt"""
        self.prompts.extend(prompts)
        print(f"✅ 已捕获 {len(prompts)} 个 Prompt")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """捕获响应"""
        if hasattr(response, 'generations') and response.generations:
            for gen in response.generations[0]:
                if hasattr(gen, 'text'):
                    self.responses.append(gen.text)


def example_save_prompt():
    """示例：保存 Prompt 供后续分析"""
    
    from sql_agent_base import SQLAgentBase
    import os
    from datetime import datetime
    
    print("\n" + "="*80)
    print("💾 保存 Prompt 到文件")
    print("="*80)
    
    # 创建 Agent
    print("\n1️⃣  创建 SQL Agent...")
    agent = SQLAgentBase()
    
    # 创建捕获器
    print("2️⃣  创建 Prompt 捕获器...")
    capturer = PromptCapture()
    
    # 执行查询
    question = "train数据集里有多少张图片"
    print(f"3️⃣  执行查询: {question}\n")
    
    try:
        result = agent.agent.invoke(
            {"input": question},
            config={"callbacks": [capturer]}
        )
        
        print("\n" + "="*80)
        print("✅ 查询完成")
        print("="*80)
        print(f"答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"\n❌ 查询错误: {str(e)}")
        return
    
    # 保存捕获的内容
    print("\n" + "="*80)
    print("💾 保存数据到文件")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_dir = os.getcwd()
    
    print(f"\n捕获了 {len(capturer.prompts)} 个 Prompt")
    print(f"捕获了 {len(capturer.responses)} 个响应")
    
    saved_files = []
    
    # 保存每个 Prompt
    for i, prompt in enumerate(capturer.prompts, 1):
        filename = f'captured_prompt_{timestamp}_{i}.txt'
        filepath = os.path.join(current_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"="*80 + "\n")
                f.write(f"Prompt #{i}\n")
                f.write(f"问题: {question}\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"长度: {len(prompt)} 字符, {prompt.count(chr(10)) + 1} 行\n")
                f.write(f"="*80 + "\n\n")
                f.write(prompt)
            
            saved_files.append(filepath)
            print(f"  ✅ Prompt {i}: {filename} ({len(prompt)} 字符)")
        except Exception as e:
            print(f"  ❌ Prompt {i}: 保存失败 - {str(e)}")
    
    # 保存响应
    if capturer.responses:
        for i, response in enumerate(capturer.responses, 1):
            filename = f'captured_response_{timestamp}_{i}.txt'
            filepath = os.path.join(current_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"="*80 + "\n")
                    f.write(f"Response #{i}\n")
                    f.write(f"问题: {question}\n")
                    f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"="*80 + "\n\n")
                    f.write(response)
                
                saved_files.append(filepath)
                print(f"  ✅ Response {i}: {filename}")
            except Exception as e:
                print(f"  ❌ Response {i}: 保存失败 - {str(e)}")
    
    # 保存完整交互记录
    summary_file = f'captured_summary_{timestamp}.txt'
    summary_path = os.path.join(current_dir, summary_file)
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"="*80 + "\n")
            f.write(f"LLM 交互完整记录\n")
            f.write(f"="*80 + "\n")
            f.write(f"问题: {question}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最终答案: {result.get('output', result)}\n")
            f.write(f"\n")
            
            for i, prompt in enumerate(capturer.prompts, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"Prompt #{i}\n")
                f.write(f"{'='*80}\n")
                f.write(prompt)
                f.write(f"\n\n")
                
                if i <= len(capturer.responses):
                    f.write(f"{'='*80}\n")
                    f.write(f"Response #{i}\n")
                    f.write(f"{'='*80}\n")
                    f.write(capturer.responses[i-1])
                    f.write(f"\n\n")
        
        saved_files.append(summary_path)
        print(f"  ✅ 完整记录: {summary_file}")
    except Exception as e:
        print(f"  ❌ 完整记录: 保存失败 - {str(e)}")
    
    # 显示保存位置
    print("\n" + "="*80)
    print("📁 文件保存位置")
    print("="*80)
    print(f"目录: {current_dir}")
    print(f"\n共保存了 {len(saved_files)} 个文件:")
    for filepath in saved_files:
        print(f"  • {os.path.basename(filepath)}")
    
    print("\n💡 提示: 你可以用文本编辑器打开这些文件查看内容")


# ============================================
# 主函数
# ============================================

def main():
    """运行示例"""
    
    print("""
选择示例:
  [1] 基本用法（打印 Prompt）
  [2] 通用用法（适用于任何 Agent）
  [3] 高级用法（保存 Prompt 到文件）
  [0] 退出
    """)
    
    choice = input("请选择 (0-3): ").strip()
    
    if choice == '1':
        example_usage()
    elif choice == '2':
        example_with_any_agent()
    elif choice == '3':
        example_save_prompt()
    elif choice == '0':
        print("👋 再见！")
    else:
        print("❌ 无效选项")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\n\n💡 提示:")
        print("如果遇到问题，请确保:")
        print("  1. 已运行 python init_database.py")
        print("  2. Ollama 服务正在运行")
        print("  3. 模型已安装（如 qwen2.5:7b）")

