"""
LLM Callbacks - 记录 LLM 的 Prompt 和 Response
提供多种回调处理器用于监控和记录 LLM 的交互过程
"""
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import os


class LLMLoggingCallback(BaseCallbackHandler):
    """
    记录 LLM 的 prompt 和 response 的回调处理器
    支持输出到控制台和文件
    """
    
    def __init__(
        self, 
        log_to_file: bool = True,
        log_file: str = "llm_interactions.log",
        log_to_console: bool = True,
        verbose: bool = True,
        save_json: bool = False,
        json_file: str = "llm_interactions.json"
    ):
        """
        初始化回调处理器
        
        Args:
            log_to_file: 是否记录到文件
            log_file: 日志文件路径
            log_to_console: 是否输出到控制台
            verbose: 是否显示详细信息
            save_json: 是否保存为JSON格式
            json_file: JSON文件路径
        """
        super().__init__()
        self.log_to_file = log_to_file
        self.log_file = log_file
        self.log_to_console = log_to_console
        self.verbose = verbose
        self.save_json = save_json
        self.json_file = json_file
        
        # 用于存储交互历史
        self.interactions = []
        
        # 计数器
        self.call_count = 0
        
        # 清空或创建日志文件（如果需要）
        if self.log_to_file and not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== LLM Interaction Log ===\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """
        LLM 开始生成时调用
        记录发送给 LLM 的 prompt
        """
        self.call_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        interaction = {
            'call_id': self.call_count,
            'timestamp': timestamp,
            'type': 'llm_start',
            'prompts': prompts,
            'model_info': serialized.get('name', 'Unknown'),
            'kwargs': {k: v for k, v in kwargs.items() if k not in ['run_id', 'parent_run_id']}
        }
        
        # 输出到控制台
        if self.log_to_console:
            self._print_to_console(interaction, is_start=True)
        
        # 写入文件
        if self.log_to_file:
            self._write_to_file(interaction, is_start=True)
        
        # 保存到历史记录
        self.interactions.append(interaction)
    
    def on_llm_end(
        self, 
        response: Any, 
        **kwargs: Any
    ) -> None:
        """
        LLM 生成结束时调用
        记录 LLM 的 response
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 提取生成的文本
        generations = []
        if hasattr(response, 'generations'):
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, 'text'):
                        generations.append(gen.text)
                    elif hasattr(gen, 'message'):
                        generations.append(str(gen.message))
        
        # 提取 token 使用情况（如果有）
        token_usage = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            if 'token_usage' in response.llm_output:
                token_usage = response.llm_output['token_usage']
        
        interaction = {
            'call_id': self.call_count,
            'timestamp': timestamp,
            'type': 'llm_end',
            'generations': generations,
            'token_usage': token_usage
        }
        
        # 更新历史记录中最后一条交互
        if self.interactions and self.interactions[-1]['call_id'] == self.call_count:
            self.interactions[-1].update({
                'response_timestamp': timestamp,
                'generations': generations,
                'token_usage': token_usage
            })
        
        # 输出到控制台
        if self.log_to_console:
            self._print_to_console(interaction, is_start=False)
        
        # 写入文件
        if self.log_to_file:
            self._write_to_file(interaction, is_start=False)
        
        # 保存为JSON
        if self.save_json:
            self._save_to_json()
    
    def on_llm_error(
        self, 
        error: Union[Exception, KeyboardInterrupt], 
        **kwargs: Any
    ) -> None:
        """
        LLM 发生错误时调用
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        error_info = {
            'call_id': self.call_count,
            'timestamp': timestamp,
            'type': 'llm_error',
            'error': str(error),
            'error_type': type(error).__name__
        }
        
        # 输出到控制台
        if self.log_to_console:
            print(f"\n{'='*80}")
            print(f"❌ LLM ERROR - Call #{self.call_count}")
            print(f"{'='*80}")
            print(f"⏰ Time: {timestamp}")
            print(f"🚨 Error Type: {error_info['error_type']}")
            print(f"📝 Error Message: {error_info['error']}")
            print(f"{'='*80}\n")
        
        # 写入文件
        if self.log_to_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"❌ LLM ERROR - Call #{self.call_count}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Time: {timestamp}\n")
                f.write(f"Error Type: {error_info['error_type']}\n")
                f.write(f"Error Message: {error_info['error']}\n")
                f.write(f"{'='*80}\n\n")
        
        # 更新历史记录
        if self.interactions and self.interactions[-1]['call_id'] == self.call_count:
            self.interactions[-1]['error'] = error_info
    
    def _print_to_console(self, interaction: Dict[str, Any], is_start: bool) -> None:
        """格式化输出到控制台"""
        if is_start:
            print(f"\n{'='*80}")
            print(f"LLM INPUT - Call #{interaction['call_id']}")
            print(f"{'='*80}")
            print(f"⏰ Time: {interaction['timestamp']}")
            print(f"🤖 Model: {interaction['model_info']}")
            
            if self.verbose and interaction.get('kwargs'):
                print(f"⚙️  Parameters: {json.dumps(interaction['kwargs'], indent=2, ensure_ascii=False)}")
            
            print(f"\n📝 Prompts ({len(interaction['prompts'])}):")
            print("-" * 80)
            for i, prompt in enumerate(interaction['prompts'], 1):
                if len(interaction['prompts']) > 1:
                    print(f"\n[Prompt {i}]")
                print(prompt)
            print("-" * 80)
        else:
            print(f"\n{'='*80}")
            print(f"📥 LLM OUTPUT - Call #{interaction['call_id']}")
            print(f"{'='*80}")
            print(f"⏰ Time: {interaction['timestamp']}")
            
            if interaction.get('token_usage'):
                print(f"💰 Token Usage: {json.dumps(interaction['token_usage'], indent=2)}")
            
            print(f"\n💬 Responses ({len(interaction['generations'])}):")
            print("-" * 80)
            for i, gen in enumerate(interaction['generations'], 1):
                if len(interaction['generations']) > 1:
                    print(f"\n[Response {i}]")
                print(gen)
            print("-" * 80)
            print()
    
    def _write_to_file(self, interaction: Dict[str, Any], is_start: bool) -> None:
        """写入到日志文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            if is_start:
                f.write(f"\n{'='*80}\n")
                f.write(f"LLM INPUT - Call # {interaction['call_id']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Time: {interaction['timestamp']}\n")
                f.write(f"Model: {interaction['model_info']}\n")
                
                if self.verbose and interaction.get('kwargs'):
                    f.write(f"Parameters: {json.dumps(interaction['kwargs'], indent=2, ensure_ascii=False)}\n")
                
                f.write(f"\nPrompts ({len(interaction['prompts'])}):\n")
                f.write("-" * 80 + "\n")
                for i, prompt in enumerate(interaction['prompts'], 1):
                    if len(interaction['prompts']) > 1:
                        f.write(f"\n[Prompt {i}]\n")
                    f.write(prompt + "\n")
                f.write("-" * 80 + "\n")
            else:
                f.write(f"\n{'='*80}\n")
                f.write(f"LLM OUTPUT - Call #{interaction['call_id']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Time: {interaction['timestamp']}\n")
                
                if interaction.get('token_usage'):
                    f.write(f"Token Usage: {json.dumps(interaction['token_usage'], indent=2)}\n")
                
                f.write(f"\nResponses ({len(interaction['generations'])}):\n")
                f.write("-" * 80 + "\n")
                for i, gen in enumerate(interaction['generations'], 1):
                    if len(interaction['generations']) > 1:
                        f.write(f"\n[Response {i}]\n")
                    f.write(gen + "\n")
                f.write("-" * 80 + "\n\n")
    
    def _save_to_json(self) -> None:
        """保存所有交互记录到JSON文件"""
        with open(self.json_file, 'a', encoding='utf-8') as f:
            json_content = json.dumps(self.interactions[-1], ensure_ascii=False)
            print(json_content, file=f)
        # with open(self.json_file, 'w', encoding='utf-8') as f:
        #     json.dump(self.interactions, f, ensure_ascii=False, indent=2)
    
    def get_interactions(self) -> List[Dict[str, Any]]:
        """获取所有交互记录"""
        return self.interactions
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_calls = len(self.interactions)
        total_prompts = sum(len(i.get('prompts', [])) for i in self.interactions)
        total_responses = sum(len(i.get('generations', [])) for i in self.interactions)
        errors = sum(1 for i in self.interactions if 'error' in i)
        
        total_tokens = 0
        if any('token_usage' in i for i in self.interactions):
            for i in self.interactions:
                if 'token_usage' in i and i['token_usage']:
                    total_tokens += i['token_usage'].get('total_tokens', 0)
        
        return {
            'total_calls': total_calls,
            'total_prompts': total_prompts,
            'total_responses': total_responses,
            'total_errors': errors,
            'total_tokens': total_tokens if total_tokens > 0 else None
        }
    
    def print_statistics(self) -> None:
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n" + "="*80)
        print("📊 LLM Interaction Statistics")
        print("="*80)
        print(f"Total Calls:     {stats['total_calls']}")
        print(f"Total Prompts:   {stats['total_prompts']}")
        print(f"Total Responses: {stats['total_responses']}")
        print(f"Total Errors:    {stats['total_errors']}")
        if stats['total_tokens']:
            print(f"Total Tokens:    {stats['total_tokens']}")
        print("="*80 + "\n")


class SimpleLLMCallback(BaseCallbackHandler):
    """
    简化版的 LLM 回调处理器
    只记录关键信息，适合快速调试
    """
    
    def __init__(self):
        super().__init__()
        self.prompts = []
        self.responses = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """记录 prompt"""
        print(f"\n🔵 LLM Start - Prompt: {prompts[0][:100]}..." if len(prompts[0]) > 100 else f"\n🔵 LLM Start - Prompt: {prompts[0]}")
        self.prompts.extend(prompts)
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """记录 response"""
        if hasattr(response, 'generations'):
            for gen_list in response.generations:
                for gen in gen_list:
                    text = gen.text if hasattr(gen, 'text') else str(gen.message)
                    print(f"🟢 LLM End - Response: {text[:100]}..." if len(text) > 100 else f"🟢 LLM End - Response: {text}")
                    self.responses.append(text)
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """记录错误"""
        print(f"🔴 LLM Error: {str(error)}")


class ChainLoggingCallback(BaseCallbackHandler):
    """
    Chain 级别的回调处理器
    记录整个 Chain 的执行过程，包括 LLM 调用
    """
    
    def __init__(self, log_file: str = "chain_interactions.log"):
        super().__init__()
        self.log_file = log_file
        self.chain_depth = 0
        
        # 初始化日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Chain Execution Log ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Chain 开始"""
        indent = "  " * self.chain_depth
        timestamp = datetime.now().strftime('%H:%M:%S')
        chain_name = serialized.get('name', 'Unknown Chain')
        
        log_msg = f"{indent}[{timestamp}] ▶️  {chain_name} START\n"
        log_msg += f"{indent}   Inputs: {json.dumps(inputs, ensure_ascii=False)[:200]}\n"
        
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
        
        self.chain_depth += 1
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Chain 结束"""
        self.chain_depth -= 1
        indent = "  " * self.chain_depth
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        log_msg = f"{indent}[{timestamp}] ✅ Chain END\n"
        log_msg += f"{indent}   Outputs: {json.dumps(outputs, ensure_ascii=False)[:200]}\n"
        
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Chain 错误"""
        self.chain_depth -= 1
        indent = "  " * self.chain_depth
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        log_msg = f"{indent}[{timestamp}] ❌ Chain ERROR: {str(error)}\n"
        
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM 开始"""
        indent = "  " * self.chain_depth
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        log_msg = f"{indent}[{timestamp}] 🤖 LLM Call\n"
        for i, prompt in enumerate(prompts):
            log_msg += f"{indent}   Prompt {i+1}: {prompt[:150]}...\n" if len(prompt) > 150 else f"{indent}   Prompt {i+1}: {prompt}\n"
        
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLM 结束"""
        indent = "  " * self.chain_depth
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        log_msg = f"{indent}[{timestamp}] 💬 LLM Response\n"
        
        if hasattr(response, 'generations'):
            for gen_list in response.generations:
                for gen in gen_list:
                    text = gen.text if hasattr(gen, 'text') else str(gen.message)
                    log_msg += f"{indent}   {text[:150]}...\n" if len(text) > 150 else f"{indent}   {text}\n"
        
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")


# 使用示例
def demo_callbacks():
    """
    演示如何使用各种回调处理器
    """
    from langchain_community.llms import Ollama
    from config import get_config
    
    config = get_config()
    
    print("="*80)
    print("🎯 LLM Callbacks 演示")
    print("="*80)
    
    # 1. 使用详细的 LLMLoggingCallback
    print("\n1️⃣ 使用 LLMLoggingCallback (详细记录)")
    print("-"*80)
    
    callback = LLMLoggingCallback(
        log_to_file=True,
        log_file="llm_detailed.log",
        log_to_console=True,
        verbose=True,
        save_json=True,
        json_file="llm_detailed.json"
    )
    
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config['ollama_base_url'],
        temperature=0,
        callbacks=[callback]
    )
    
    # 测试调用
    response = llm.invoke("什么是SQL？请用一句话回答。")
    print(f"\n最终响应: {response}")
    
    # 显示统计信息
    callback.print_statistics()
    
    # 2. 使用简化的 SimpleLLMCallback
    print("\n2️⃣ 使用 SimpleLLMCallback (简化记录)")
    print("-"*80)
    
    simple_callback = SimpleLLMCallback()
    
    llm2 = Ollama(
        model=config['ollama_model'],
        base_url=config['ollama_base_url'],
        temperature=0,
        callbacks=[simple_callback]
    )
    
    response2 = llm2.invoke("什么是数据库？请用一句话回答。")
    print(f"\n总共记录了 {len(simple_callback.prompts)} 个 prompts")
    print(f"总共记录了 {len(simple_callback.responses)} 个 responses")
    
    # 3. 使用 ChainLoggingCallback
    print("\n3️⃣ 使用 ChainLoggingCallback (Chain 级别记录)")
    print("-"*80)
    
    chain_callback = ChainLoggingCallback(log_file="chain_execution.log")
    
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent
    from langchain.agents.agent_types import AgentType
    
    db = SQLDatabase.from_uri(config['database_uri'])
    
    agent = create_sql_agent(
        llm=Ollama(
            model=config['ollama_model'],
            base_url=config['ollama_base_url'],
            temperature=0,
            callbacks=[chain_callback]
        ),
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        callbacks=[chain_callback]
    )
    
    result = agent.invoke({"input": "有多少个部门？"})
    print(f"\n最终结果: {result['output']}")
    
    print("\n" + "="*80)
    print("✅ 演示完成！查看生成的日志文件:")
    print("   - llm_detailed.log")
    print("   - llm_detailed.json")
    print("   - chain_execution.log")
    print("="*80)


if __name__ == '__main__':
    demo_callbacks()

