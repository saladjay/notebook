"""
自定义的ReAct输出解析器，兼容Qwen3的思考模式
保留原始输出用于callback记录，只在解析时移除<think>标签
"""
import re
from typing import Union
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import logging

logger = logging.getLogger(__name__)
class QwenReActOutputParser(AgentOutputParser):
    """
    自定义解析器，能够处理Qwen3模型的<think>标签
    
    重要特性：
    - 在log中保留原始输出（包括<think>标签），供callback记录使用
    - 只在解析Action/Answer时移除<think>标签
    """
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        解析LLM输出，提取Action和Action Input
        
        Args:
            text: LLM的原始输出文本（可能包含<think>标签）
            
        Returns:
            AgentAction或AgentFinish对象
            注意：log字段保存清理后的文本（避免<think>污染下一轮对话）
                 原始文本通过Callback在解析前捕获
        """
        # 移除<think>标签
        cleaned_text = self._remove_thinking_tags(text)
        
        # 检查是否包含Final Answer
        if "Final Answer:" in cleaned_text or "最终答案:" in cleaned_text:
            return self._parse_final_answer(cleaned_text)
        

        # 解析Action和Action Input
        return self._parse_action(cleaned_text)
    
    def _remove_thinking_tags(self, text: str) -> str:
        """
        移除<think>...</think>标签及其内容
        使用栈匹配算法：保留成对的标签内容，删除不成对的标签
        """
        # 使用栈匹配算法删除不成对的标签
        positions = []
        
        # 查找所有 <think> 标签位置
        pos = 0
        while True:
            pos = text.find('<think>', pos)
            if pos == -1:
                break
            positions.append(('open', pos, pos + len('<think>')))
            pos += 1
        
        # 查找所有 </think> 标签位置
        pos = 0
        while True:
            pos = text.find('</think>', pos)
            if pos == -1:
                break
            positions.append(('close', pos, pos + len('</think>')))
            pos += 1
        
        # 按位置排序
        positions.sort(key=lambda x: x[1])
        
        # 使用栈匹配，找出成对的标签
        stack = []
        paired_ranges = []  # 成对的标签范围
        unpaired_tags = []  # 不成对的标签位置
        
        for tag_type, start, end in positions:
            if tag_type == 'open':
                stack.append(('open', start, end))
            else:  # 'close'
                if stack and stack[-1][0] == 'open':
                    # 配对成功
                    open_tag = stack.pop()
                    paired_ranges.append((open_tag[1], end))  # 记录配对范围
                else:
                    # 没有匹配的开始标签，标记为不成对
                    unpaired_tags.append((start, end))
        
        # 栈中剩余的都是没有配对的开始标签
        for tag_type, start, end in stack:
            unpaired_tags.append((start, end))
        
        # 移除所有成对的 <think>...</think> 内容
        for start, end in reversed(sorted(paired_ranges)):
            text = text[:start] + text[end:]
        
        # 移除所有不成对的标签
        for start, end in reversed(sorted(unpaired_tags)):
            text = text[:start] + text[end:]
        
        # 清理多余的空白
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _parse_final_answer(self, cleaned_text: str) -> AgentFinish:
        """
        解析最终答案
        
        Args:
            cleaned_text: 移除<think>标签后的文本
        """
        # 支持中英文
        if "Final Answer:" in cleaned_text:
            final_answer = cleaned_text.split("Final Answer:")[-1].strip()
        else:
            final_answer = cleaned_text.split("最终答案:")[-1].strip()
        
        return AgentFinish(
            return_values={"output": final_answer},
            log=cleaned_text  # 保存清理后的文本，避免污染下一轮对话
        )
    
    def _parse_action(self, cleaned_text: str) -> AgentAction:
        """
        解析Action和Action Input
        
        Args:
            cleaned_text: 移除<think>标签后的文本
        """
        logger.info(f"解析Action和Action Input: {cleaned_text}")
        clean_text = cleaned_text.replace("Observation", "")
        # 尝试匹配标准格式: Action: xxx\nAction Input: yyy
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', cleaned_text, re.IGNORECASE)
        action_input_match = re.search(r'Action Input:\s*(.*?)(?:\n|$)', cleaned_text, re.DOTALL | re.IGNORECASE)
        
        # 也支持中文格式
        if not action_match:
            action_match = re.search(r'动作:\s*(.+?)(?:\n|$)', cleaned_text)
        if not action_input_match:
            action_input_match = re.search(r'动作输入:\s*(.+?)(?:\n|$)', cleaned_text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            # 仅在确认为包裹型引号时去壳，避免误删SQL末尾的引号
            if (len(action_input) >= 2 and (
                (action_input[0] == '"' and action_input[-1] == '"') or
                (action_input[0] == "'" and action_input[-1] == "'")
            )):
                action_input = action_input[1:-1]
            
            return AgentAction(
                tool=action,
                tool_input=action_input,
                log=cleaned_text  # 保存清理后的文本，避免污染下一轮对话
            )
        
        # 如果无法解析，抛出异常
        raise OutputParserException(
            f"Could not parse LLM output after removing thinking tags: `{cleaned_text}`"
        )
    
    @property
    def _type(self) -> str:
        return "qwen-react-single-input"


def get_qwen_react_parser():
    """获取Qwen兼容的ReAct解析器"""
    return QwenReActOutputParser()

