"""
LLM Wrapper - 记录和改写模型输入输出的包装器
支持多种 LLM 后端，提供灵活的输入输出处理机制
继承自 BaseLanguageModel 以兼容 LangChain 的类型检查
"""
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import json
import os
import traceback
from abc import ABC, abstractmethod
import re

class LLMInterceptor(ABC):
    """
    LLM 拦截器基类
    可以继承此类来实现自定义的输入输出处理逻辑
    """
    
    @abstractmethod
    def before_call(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """
        在调用 LLM 前执行
        
        Args:
            prompt: 原始输入提示词
            **kwargs: 其他参数
            
        Returns:
            tuple[str, dict]: (修改后的提示词, 修改后的参数)
        """
        pass
    
    @abstractmethod
    def after_call(self, response: Any, **kwargs) -> Any:
        """
        在 LLM 返回后执行
        
        Args:
            response: LLM 的原始响应
            **kwargs: 其他参数
            
        Returns:
            Any: 修改后的响应
        """
        pass
    
    def on_error(self, error: Exception, **kwargs) -> None:
        """
        当 LLM 调用出错时执行
        
        Args:
            error: 异常对象
            **kwargs: 其他参数
        """
        pass


class LoggingInterceptor(LLMInterceptor):
    """
    日志记录拦截器
    记录所有的输入和输出到文件和/或控制台
    """
    
    def __init__(
        self, 
        json_file: str = "llm_wrapper_history.jsonl",
        verbose: bool = True
    ):

        self.save_json = True
        self.json_file = json_file
        self.verbose = verbose
        self.call_count = 0
    
    def before_call(self, prompt: str, *args, **kwargs) -> tuple[str, dict]:
        """记录输入"""
        self.call_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        log_data = {
            'call_id': self.call_count,
            'timestamp': timestamp,
            'type': 'input',
            'prompt': prompt if isinstance(prompt, str) else str(prompt.text),
            'kwargs': kwargs
        }
        
        # 保存原始输入信息供 after_call 使用
        self._current_log_data = log_data
        
        return prompt, args, kwargs
    
    def after_call(self, response: Any, **kwargs) -> Any:
        """记录输出"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # 提取响应文本
        response_text = self._extract_response_text(response)
        
        log_data = {
            'call_id': self.call_count,
            'timestamp': timestamp,
            'type': 'output',
            'response': response_text,
            'response_type': type(response).__name__
        }
        
        # 保存完整交互到 JSON
        if self.save_json and hasattr(self, '_current_log_data'):
            full_interaction = {
                **self._current_log_data,
                'response': response_text,
                'response_timestamp': timestamp,
                'response_type': type(response).__name__
            }

            with open(self.json_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(full_interaction, ensure_ascii=False) + '\n')
        
        return response
    
    def on_error(self, error: Exception, **kwargs) -> None:
        """记录错误"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        error_msg = f"{type(error).__name__}: {str(error)}"
        error_trace = traceback.format_exc()
        
        
        print(f"\n{'='*80}")
        print(f"❌ LLM ERROR #{self.call_count}")
        print(f"{'='*80}")
        print(f"⏰ Time: {timestamp}")
        print(f"🚨 Error: {error_msg}")
        if self.verbose:
            print(f"\n📋 Traceback:\n{error_trace}")
        print(f"{'='*80}\n")
    
    def _extract_response_text(self, response: Any) -> str:
        """提取响应文本"""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        else:
            return str(response)


class PromptTemplateInterceptor(LLMInterceptor):
    """
    提示词模板拦截器
    在原始提示词前后添加固定内容
    """
    
    def __init__(
        self, 
        prefix: str = "",
        suffix: str = "",
        template: Optional[str] = None
    ):
        """
        Args:
            prefix: 在提示词前添加的内容
            suffix: 在提示词后添加的内容
            template: 模板字符串，使用 {prompt} 作为占位符
        """
        self.prefix = prefix
        self.suffix = suffix
        self.template = template
    
    def before_call(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """添加前缀和后缀"""
        if self.template:
            modified_prompt = self.template.format(prompt=prompt)
        else:
            modified_prompt = self.prefix + prompt + self.suffix
        
        return modified_prompt, kwargs
    
    def after_call(self, response: Any, **kwargs) -> Any:
        """不修改输出"""
        return response


class ResponseCleanerInterceptor(LLMInterceptor):
    """
    响应清理拦截器
    清理和规范化 LLM 的输出
    """
    
    def __init__(
        self,
        strip_whitespace: bool = True,
        remove_tags: List[str] = None,
        replace_patterns: Dict[str, str] = None
    ):
        """
        Args:
            strip_whitespace: 是否去除首尾空白
            remove_tags: 要移除的 XML/HTML 标签列表，如 ['<think>', '</think>']
            replace_patterns: 字符串替换规则，如 {'old': 'new'}
        """
        self.strip_whitespace = strip_whitespace
        self.remove_tags = remove_tags or []
        self.replace_patterns = replace_patterns or {}
    
    def before_call(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """不修改输入"""
        return prompt, kwargs
    
    def after_call(self, response: Any, **kwargs) -> Any:
        """清理输出"""
        
        # 提取文本
        if isinstance(response, str):
            text = response
        elif hasattr(response, 'content'):
            text = response.content
        elif hasattr(response, 'text'):
            text = response.text
        else:
            return response
        # with open(r"generate_data\test.json", 'a', encoding='utf-8') as f:
        #     print(f"before text: {text}", file=f)
        # 应用清理规则
        if self.strip_whitespace:
            text = text.strip()
        
        # 移除<think>和</think>标签之间的内容，包括换行符
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # 移除标签
        for tag in self.remove_tags:
            text = text.replace(tag, '')
        
        # 替换模式
        for old, new in self.replace_patterns.items():
            text = text.replace(old, new)
        
        # with open(r"generate_data\test.json", 'a', encoding='utf-8') as f:
        #     print(f"after text: {text}", file=f)
        # 更新响应对象
        if isinstance(response, str):
            return text
        elif hasattr(response, 'content'):
            response.content = text
            return response
        elif hasattr(response, 'text'):
            response.text = text
            return response
        else:
            return text


class Qwen3ReActInterceptor(LLMInterceptor):
    """
    Qwen3 ReAct 拦截器
    清理 Qwen3 的输出
    """
    
    def __init__(self,
        jsonl_file: str = "qwen3_react_history.jsonl"
    ):
        self.jsonl_file = jsonl_file
        self.call_count = 0
        self.remove_tags = ['<think>', '</think>']
    
    def before_call(self, prompt: str, *args, **kwargs) -> tuple[str, dict]:
        """记录输入"""
        self.call_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        log_data = {
            'call_id': self.call_count,
            'timestamp': timestamp,
            'type': 'input',
            'prompt': self._extract_text(prompt),
            # 'args': args,
            # 'kwargs': kwargs
        }
        
        # 保存原始输入信息供 after_call 使用
        self._current_log_data = log_data
        
        return prompt, args, kwargs

    def after_call(self, response: Any, *args, **kwargs) -> Any:
        """记录输出"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        response = self._clean_response(response)
        # 提取响应文本
        response_text = self._extract_text(response)
        
        # 保存完整交互到 JSON
        if hasattr(self, '_current_log_data'):
            full_interaction = {
                **self._current_log_data,
                'modified_response': response_text,
                'response_timestamp': timestamp,
                'response_type': type(response).__name__
            }

            with open(self.jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(full_interaction, ensure_ascii=False) + '\n')
        
        return response
    
    def on_error(self, error: Exception, **kwargs) -> None:
        """记录错误"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        error_msg = f"{type(error).__name__}: {str(error)}"
        error_trace = traceback.format_exc()
        
        
        print(f"\n{'='*80}")
        print(f"❌ LLM ERROR #{self.call_count}")
        print(f"{'='*80}")
        print(f"⏰ Time: {timestamp}")
        print(f"🚨 Error: {error_msg}")
        print(f"\n📋 Traceback:\n{error_trace}")
        print(f"{'='*80}\n")
    
    def _extract_text(self, response: Any) -> str:
        """提取响应文本"""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        else:
            return str(response)

    def _clean_response(self, response: Any) -> Any:
        """清理输出"""
        
        # 提取文本
        if isinstance(response, str):
            text = response
        elif hasattr(response, 'content'):
            text = response.content
        elif hasattr(response, 'text'):
            text = response.text
        else:
            return response
        if hasattr(self, '_current_log_data'):
            self._current_log_data['response'] = text
        # 应用清理规
        text = text.strip()
        
        # 移除<think>和</think>标签之间的内容，包括换行符
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL)

        # 移除Observation:
        text = text.replace("Observation:", "")
        # 移除标签
        for tag in self.remove_tags:
            text = text.replace(tag, '')

        # 更新响应对象
        if isinstance(response, str):
            return text
        elif hasattr(response, 'content'):
            response.content = text
            return response
        elif hasattr(response, 'text'):
            response.text = text
            return response
        else:
            return text
    
    

class LLMWrapper(BaseLanguageModel):
    """
    LLM 包装器
    支持多个拦截器的链式调用
    继承自 BaseLanguageModel 以兼容 LangChain 的类型检查
    """
    
    def __init__(
        self,
        llm: Any,
        interceptors: Optional[List[LLMInterceptor]] = None,
        enable_default_logging: bool = False
    ):
        """
        Args:
            llm: 底层的 LLM 实例（LangChain LLM 或其他）
            interceptors: 拦截器列表
            enable_default_logging: 是否启用默认的日志记录拦截器
        """
        super().__init__()  # 调用父类初始化
        self._llm = llm
        self._interceptors = interceptors or []
        
        # 如果启用默认日志记录且没有日志拦截器，则添加一个
        if enable_default_logging and not any(isinstance(i, LoggingInterceptor) for i in self._interceptors):
            self._interceptors.insert(0, LoggingInterceptor())
    
    def add_interceptor(self, interceptor: LLMInterceptor) -> 'LLMWrapper':
        """添加拦截器"""
        self._interceptors.append(interceptor)
        return self
    
    def invoke(self, prompt: str, *args, **kwargs) -> Any:
        """
        调用 LLM
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            Any: LLM 的响应
        """
        try:
            # 执行所有 before_call 拦截器
            modified_prompt = prompt
            modified_kwargs = kwargs.copy()
            
            for interceptor in self._interceptors:

                modified_prompt, args, modified_kwargs = interceptor.before_call(
                    modified_prompt, *args, **modified_kwargs
                )
            
            # 调用底层 LLM
            response = self._llm.invoke(modified_prompt, *args, **modified_kwargs)
            
            # 执行所有 after_call 拦截器（逆序）
            modified_response = response
            for interceptor in reversed(self._interceptors):
                modified_response = interceptor.after_call(
                    modified_response, 
                    original_prompt=prompt,
                    modified_prompt=modified_prompt
                )
            
            return modified_response
            
        except Exception as e:
            # 通知所有拦截器发生错误
            for interceptor in self._interceptors:
                interceptor.on_error(e, prompt=prompt, **kwargs)
            raise
    
    def __call__(self, prompt: str, **kwargs) -> Any:
        """支持直接调用"""
        return self.invoke(prompt, **kwargs)
    
    # ========================================
    # 实现 BaseLanguageModel 的必需方法
    # ========================================
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        实现 BaseLanguageModel 的 _generate 方法
        这是 LangChain 内部调用的核心方法
        """
        # 对每个 prompt 应用拦截器
        results = []
        for prompt in prompts:
            try:
                # 执行所有 before_call 拦截器
                modified_prompt = prompt
                modified_kwargs = kwargs.copy()
                
                for interceptor in self._interceptors:
                    modified_prompt, modified_kwargs = interceptor.before_call(
                        modified_prompt, **modified_kwargs
                    )
                
                # 调用底层 LLM 的 _generate 方法
                if hasattr(self._llm, '_generate'):
                    llm_result = self._llm._generate(
                        [modified_prompt], 
                        stop=stop, 
                        run_manager=run_manager,
                        **modified_kwargs
                    )
                else:
                    # 如果底层 LLM 没有 _generate，尝试使用 invoke
                    response = self._llm.invoke(modified_prompt, **modified_kwargs)
                    # 构造 LLMResult
                    llm_result = LLMResult(generations=[[Generation(text=str(response))]])
                
                # 执行所有 after_call 拦截器（逆序）
                # 注意：这里处理的是生成的文本
                for generation_list in llm_result.generations:
                    for generation in generation_list:
                        modified_response = generation.text
                        for interceptor in reversed(self._interceptors):
                            modified_response = interceptor.after_call(
                                modified_response,
                                original_prompt=prompt,
                                modified_prompt=modified_prompt
                            )
                        generation.text = modified_response
                
                results.append(llm_result)
                
            except Exception as e:
                # 通知所有拦截器发生错误
                for interceptor in self._interceptors:
                    interceptor.on_error(e, prompt=prompt, **kwargs)
                raise
        
        # 合并所有结果
        if len(results) == 1:
            return results[0]
        else:
            # 合并多个结果
            all_generations = []
            for result in results:
                all_generations.extend(result.generations)
            return LLMResult(generations=all_generations)
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        if hasattr(self._llm, '_llm_type'):
            return f"wrapped_{self._llm._llm_type}"
        return "llm_wrapper"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数"""
        if hasattr(self._llm, '_identifying_params'):
            return self._llm._identifying_params
        return {"llm_type": self._llm_type}
    
    # 转发其他可能需要的属性和方法到底层 LLM
    def __getattr__(self, name: str) -> Any:
        """转发未定义的属性到底层 LLM"""
        # 避免无限递归
        if name in ['llm', 'interceptors']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._llm, name)

    def agenerate_prompt(self, prompt: str, *args, **kwargs) -> Any:
        return self._llm.generate_prompt(prompt, *args, **kwargs)

    def apredict(self, prompt: str, *args, **kwargs) -> Any:
        return self._llm.predict(prompt, *args, **kwargs)

    def apredict_messages(self, messages: List[BaseMessage], *args, **kwargs) -> Any:
        return self._llm.predict_messages(messages, *args, **kwargs)

    def generate_prompt(self, prompt: str, *args, **kwargs) -> Any:
        return self._llm.generate_prompt(prompt, *args, **kwargs)

    def predict(self, prompt: str, *args, **kwargs) -> Any:
        return self._llm.predict(prompt, *args, **kwargs)

    def predict_messages(self, messages: List[BaseMessage], *args, **kwargs) -> Any:
        return self._llm.predict_messages(messages, *args, **kwargs)

# 便捷工厂函数



# 使用示例
if __name__ == '__main__':
    print("="*80)
    print("🎯 LLM Wrapper 使用示例")
    print("="*80)
    
    # 示例1: 基础日志记录
    print("\n📝 示例1: 基础日志记录")
    print("-"*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        # 创建原始 LLM
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 创建包装后的 LLM
        wrapped_llm = create_logging_wrapper(
            llm,
            log_file="example_llm_wrapper.log",
            json_file="example_llm_wrapper.jsonl"
        )
        
        # 调用
        response = wrapped_llm("什么是 SQL？请用一句话回答。")
        print(f"\n✅ 最终响应: {response}")
        
    except ImportError as e:
        print(f"⚠️ 无法导入必要的库: {e}")
        print("这是一个演示示例，需要安装 langchain_ollama 库")
    
    # 示例2: 自定义拦截器
    print("\n\n📝 示例2: 自定义系统提示词和输出清理")
    print("-"*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        # 创建原始 LLM
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 创建自定义包装器
        wrapped_llm = create_custom_wrapper(
            llm,
            system_prompt="你是一个专业的数据库专家。请用简洁专业的语言回答。",
            remove_think_tags=True,
            log_interactions=True
        )
        
        # 调用
        response = wrapped_llm("什么是数据库索引？")
        print(f"\n✅ 最终响应: {response}")
        
    except ImportError as e:
        print(f"⚠️ 无法导入必要的库: {e}")
    
    # 示例3: 手动组合拦截器
    print("\n\n📝 示例3: 手动组合多个拦截器")
    print("-"*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        # 创建原始 LLM
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 手动创建拦截器链
        wrapper = LLMWrapper(llm, interceptors=[
            LoggingInterceptor(
                log_file="custom_log.log",
                log_to_console=True,
                verbose=True
            ),
            PromptTemplateInterceptor(
                template="作为一名资深工程师，请回答：{prompt}\n\n请提供详细且准确的答案。"
            ),
            ResponseCleanerInterceptor(
                strip_whitespace=True,
                remove_tags=['<think>', '</think>'],
                replace_patterns={
                    '...': '。',
                    '！！': '！'
                }
            )
        ])
        
        # 调用
        response = wrapper("什么是事务？")
        print(f"\n✅ 最终响应: {response}")
        
    except ImportError as e:
        print(f"⚠️ 无法导入必要的库: {e}")
    
    print("\n" + "="*80)
    print("✅ 示例完成！查看生成的日志文件:")
    print("   - example_llm_wrapper.log")
    print("   - example_llm_wrapper.jsonl")
    print("   - custom_log.log")
    print("="*80)

