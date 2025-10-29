"""
Qwen原始输出捕获Callback
在解析前捕获完整的LLM输出（包含<think>标签），用于数据收集和分析
"""
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import json
import logging
import re
from datetime import datetime


class QwenOutputCaptureCallback(BaseCallbackHandler):
    """
    捕获Qwen模型的原始输出（包含<think>标签）
    
    工作原理：
    1. 在LLM生成完成后（on_llm_end），立即捕获原始输出
    2. 此时输出还未经过解析器处理，包含完整的<think>标签
    3. 保存到日志文件或内存，供后续分析使用
    
    这样即使解析器清理了log字段，我们仍能获取完整数据
    """
    
    def __init__(
        self,
        log_file: str = None,
        console_output: bool = False,
        save_to_memory: bool = True
    ):
        """
        初始化Callback
        
        Args:
            log_file: 日志文件路径（可选）
            console_output: 是否输出到控制台
            save_to_memory: 是否保存到内存（用于后续访问）
        """
        super().__init__()
        self.log_file = log_file
        self.console_output = console_output
        self.save_to_memory = save_to_memory
        
        # 内存存储
        self.captured_outputs = [] if save_to_memory else None
        
        # 设置日志
        if log_file:
            self.logger = logging.getLogger("QwenOutputCapture")
            self.logger.setLevel(logging.INFO)
            
            # 避免重复添加handler
            if not self.logger.handlers:
                handler = logging.FileHandler(log_file, encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = None
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        """
        LLM生成完成时的回调
        
        关键时机：此时输出刚生成，还未经过解析器处理
        """
        # 提取原始输出
        if response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    original_text = generation.text
                    
                    # 检查是否包含<think>标签
                    has_think = '<think>' in original_text
                    
                    # 构建记录
                    record = {
                        "timestamp": self._get_timestamp(),
                        "original_output": original_text,
                        "has_think_tags": has_think,
                        "output_length": len(original_text)
                    }
                    
                    # 保存到内存
                    if self.save_to_memory and self.captured_outputs is not None:
                        self.captured_outputs.append(record)
                    
                    # 写入日志文件
                    if self.logger:
                        self.logger.info(json.dumps(record, ensure_ascii=False))
                    
                    # 控制台输出
                    if self.console_output:
                        print("\n" + "=" * 80)
                        print("🔍 捕获原始LLM输出")
                        print("=" * 80)
                        if has_think:
                            print("✅ 包含<think>标签")
                        else:
                            print("⚠️  不包含<think>标签")
                        print(f"输出长度: {len(original_text)} 字符")
                        print("-" * 80)
                        print(original_text[:500])  # 显示前500字符
                        if len(original_text) > 500:
                            print("...")
                        print("=" * 80)
    
    def get_captured_outputs(self) -> List[Dict]:
        """获取所有捕获的输出"""
        return self.captured_outputs if self.captured_outputs else []
    
    def get_last_output(self) -> Dict:
        """获取最后一次捕获的输出"""
        if self.captured_outputs and len(self.captured_outputs) > 0:
            return self.captured_outputs[-1]
        return None
    
    def clear_outputs(self):
        """清空内存中的输出"""
        if self.captured_outputs is not None:
            self.captured_outputs.clear()
    
    @staticmethod
    def _get_timestamp():
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CombinedQwenCallback(BaseCallbackHandler):
    """
    组合Callback：同时捕获原始输出和Agent交互
    
    使用场景：既要记录完整的<think>内容，又要记录Agent的执行流程
    """
    
    def __init__(
        self,
        log_file: str = "qwen_full_log.jsonl",
        console_output: bool = False
    ):
        """
        初始化组合Callback
        
        Args:
            log_file: 日志文件路径
            console_output: 是否输出到控制台
        """
        super().__init__()
        self.log_file = log_file
        self.console_output = console_output
        self.session_data = {
            "llm_prompts": [],
            "llm_outputs": [],
            "agent_actions": [],
            "tool_executions": [],
            "chains": []
        }
        self.call_count = 0
        self.interactions = []
        self.chain_count = 0
        self.chain_depth = 0  # 跟踪 Chain 嵌套深度
        self.current_question_data = {
            "prompts": [],
            "responses": [],
            "started_at": None,
            "ended_at": None
        }
    
    @staticmethod
    def _clean_unpaired_think_tags(text: str) -> str:
        """
        使用栈匹配算法清理不成对的<think>标签
        保留成对的标签及其内容，只删除不成对的标签
        
        这是 LangChain 机制的一部分：在 callback 中处理 LLM 输出
        """
        if not isinstance(text, str):
            return text
        
        # 查找所有标签位置
        positions = []
        
        # 查找所有 <think>
        pos = 0
        while True:
            pos = text.find('<think>', pos)
            if pos == -1:
                break
            positions.append(('open', pos, pos + len('<think>')))
            pos += 1
        
        # 查找所有 </think>
        pos = 0
        while True:
            pos = text.find('</think>', pos)
            if pos == -1:
                break
            positions.append(('close', pos, pos + len('</think>')))
            pos += 1
        
        # 如果没有标签，直接返回
        if not positions:
            return text
        
        # 按位置排序
        positions.sort(key=lambda x: x[1])
        
        # 使用栈匹配
        stack = []
        to_remove = []  # 需要删除的不成对标签位置
        
        for tag_type, start, end in positions:
            if tag_type == 'open':
                stack.append(('open', start, end))
            else:  # 'close'
                if stack and stack[-1][0] == 'open':
                    # 配对成功，弹出
                    stack.pop()
                else:
                    # 没有匹配的开始标签，标记删除
                    to_remove.append((start, end))
        
        # 栈中剩余的都是没有配对的开始标签
        for tag_type, start, end in stack:
            to_remove.append((start, end))
        
        # 如果没有需要删除的，直接返回
        if not to_remove:
            return text
        
        # 按位置倒序删除（避免位置偏移）
        to_remove.sort(reverse=True)
        
        result = text
        for start, end in to_remove:
            result = result[:start] + result[end:]
        
        # 清理多余的空白
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        return result

    def on_chain_start(self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any) -> None:
        """记录Chain开始"""
        self.chain_depth += 1
        self.chain_count += 1
        
        # 如果是最外层 Chain（AgentExecutor），记录问题开始
        if self.chain_depth == 1:
            self.current_question_data["started_at"] = self._get_timestamp()
            self.current_question_data["prompts"] = []
            self.current_question_data["responses"] = []
            if self.console_output:
                print(f"\n{'='*80}")
                print(f"🎯 新问题开始")
                print(f"{'='*80}\n")
        
        self.session_data["chains"].append({
            "chain_id": self.chain_count,
            "chain_depth": self.chain_depth,
            "timestamp": self._get_timestamp(),
            "inputs": inputs,
            "serialized_name": serialized.get('name', 'Unknown') if serialized else 'Unknown'
        })
        if self.console_output:
            print(f"🔗 Chain {self.chain_count} 开始 (深度: {self.chain_depth})\n")

    def on_llm_start(self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any) -> None:
        """
        LLM 开始生成时调用
        记录发送给 LLM 的 prompt
        """
        self.call_count += 1
        timestamp = self._get_timestamp()
        
        interaction = {
            'call_id': self.call_count,
            'timestamp': timestamp,
            'type': 'llm_start',
            'prompts': prompts,
            'model_info': serialized.get('name', 'Unknown') if serialized else 'Unknown',
            'kwargs': {k: v for k, v in kwargs.items() if k not in ['run_id', 'parent_run_id']}
        }
        # 保存到历史记录
        self.interactions.append(interaction)
        self.session_data["llm_prompts"].append({
            'prompts': prompts, 
            'call_id': self.call_count,
            'timestamp': timestamp
        })
        
        # 收集当前问题的 prompts
        self.current_question_data["prompts"].extend(prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        捕获LLM原始输出并清理不成对的<think>标签
        
        🔧 LangChain机制修复点：在 callback 中拦截 LLM 输出，立即清理不成对标签
        这样保证后续所有环节（parser、记录、保存）都使用干净的数据
        """
        generations = []
        if response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, 'text'):
                        # ✅ 关键：在捕获时就清理不成对的标签
                        clean_text = self._clean_unpaired_think_tags(generation.text)
                        generations.append(clean_text)
                    elif hasattr(generation, 'message'):
                        clean_text = self._clean_unpaired_think_tags(str(generation.message))
                        generations.append(clean_text)

        # 将所有生成的文本合并为一个字符串
        output_text = '\n'.join(generations) if generations else ''
        
        # 更新历史记录中最后一条交互
        if self.interactions and self.interactions[-1]['call_id'] == self.call_count:
            self.interactions[-1].update({
                'response_timestamp': self._get_timestamp(),
                'generations': generations,
                'token_usage': response.llm_output.get('token_usage', {}) if response.llm_output else {}
            })

        self.session_data["llm_outputs"].append({
            "timestamp": self._get_timestamp(),
            "output": output_text,
            "has_think": '<think>' in output_text,
            'call_id': self.call_count
        })
        
        # 收集当前问题的 responses
        self.current_question_data["responses"].append({
            "call_id": self.call_count,
            "timestamp": self._get_timestamp(),
            "output": output_text,
            "has_think": '<think>' in output_text
        })
        
        if self.log_file:
            self.save_session()

    
    def on_agent_action(self, action, **kwargs: Any) -> None:
        """记录Agent动作"""
        self.session_data["agent_actions"].append({
            "timestamp": self._get_timestamp(),
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log  # 注意：这里是清理后的log
        })
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """记录工具执行结果"""
        self.session_data["tool_executions"].append({
            "timestamp": self._get_timestamp(),
            "output": output
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """
        Chain 结束时调用
        当深度为 1 时，表示最外层 AgentExecutor 完成，即一个完整问题的结束
        """
        self.chain_depth -= 1
        
        # 如果是最外层 Chain 结束，表示一个问题完成
        if self.chain_depth == 0:
            self.current_question_data["ended_at"] = self._get_timestamp()
            
            if self.console_output:
                print(f"\n{'='*80}")
                print(f"✅ 问题完成")
                print(f"   开始时间: {self.current_question_data['started_at']}")
                print(f"   结束时间: {self.current_question_data['ended_at']}")
                print(f"   Prompt 次数: {len(self.current_question_data['prompts'])}")
                print(f"   Response 次数: {len(self.current_question_data['responses'])}")
                print(f"{'='*80}\n")
    
    def get_current_question_data(self) -> Dict[str, Any]:
        """
        获取当前问题的完整 prompt 和 response 数据
        
        Returns:
            包含所有 prompts 和 responses 的字典
        """
        return {
            "started_at": self.current_question_data["started_at"],
            "ended_at": self.current_question_data["ended_at"],
            "prompts": self.current_question_data["prompts"],
            "responses": self.current_question_data["responses"],
            "total_llm_calls": len(self.current_question_data["responses"])
        }
    
    def save_session(self):
        """保存会话数据到文件"""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json_content = json.dumps(self.interactions[-1], ensure_ascii=False)
                print(json_content, file=f)
            
    
    @staticmethod
    def _get_timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_qwen_callbacks(
    original_output_file: str = "qwen_original_outputs.log",
    full_session_file: str = "qwen_full_session.json",
    console_output: bool = False
) -> List[BaseCallbackHandler]:
    """
    创建Qwen专用的Callback组合
    
    Returns:
        [原始输出捕获Callback, 完整会话Callback]
    """
    return [
        QwenOutputCaptureCallback(
            log_file=original_output_file,
            console_output=console_output,
            save_to_memory=True
        ),
        CombinedQwenCallback(
            log_file=full_session_file,
            console_output=console_output
        )
    ]

