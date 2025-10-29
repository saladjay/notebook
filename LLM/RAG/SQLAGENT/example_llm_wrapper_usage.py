"""
LLM Wrapper 使用示例
演示如何在实际项目中使用 LLM Wrapper 来记录和改写模型输入输出
"""

from llm_wrapper import (
    LLMWrapper,
    LoggingInterceptor,
    PromptTemplateInterceptor,
    ResponseCleanerInterceptor,
    LLMInterceptor,
    create_logging_wrapper,
    create_custom_wrapper
)
from typing import Any


# ============================================================================
# 示例 1: 基础使用 - 简单的日志记录
# ============================================================================

def example_basic_logging():
    """基础使用：只记录输入输出"""
    print("\n" + "="*80)
    print("📝 示例 1: 基础日志记录")
    print("="*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        # 1. 创建原始 LLM
        llm = OllamaLLM(
            model=OllamaModel,
            base_url=OllamaURI,
            temperature=0.7
        )
        
        # 2. 使用工厂函数创建带日志的包装器
        wrapped_llm = create_logging_wrapper(
            llm,
            log_file="logs/basic_example.log",
            json_file="logs/basic_example.jsonl",
            log_to_console=True,
            verbose=True
        )
        
        # 3. 使用包装后的 LLM（完全兼容原始 LLM 的接口）
        response = wrapped_llm.invoke("什么是SQL？请用一句话回答。")
        
        print(f"\n✅ 最终响应: {response}")
        print(f"📄 日志已保存到: logs/basic_example.log")
        print(f"📊 JSON 历史已保存到: logs/basic_example.jsonl")
        
    except ImportError as e:
        print(f"⚠️ 请先安装依赖: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")


# ============================================================================
# 示例 2: 自定义拦截器 - 添加系统提示词
# ============================================================================

def example_custom_prompt():
    """添加自定义系统提示词"""
    print("\n" + "="*80)
    print("📝 示例 2: 自定义系统提示词")
    print("="*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 创建自定义提示词拦截器
        system_prompt = """你是一个专业的数据库专家。
请遵循以下规则回答问题：
1. 使用简洁专业的语言
2. 提供具体的例子
3. 如果涉及代码，使用代码块格式
4. 不要使用 XML 标签（如 <think>）进行内部思考
"""
        
        wrapped_llm = LLMWrapper(
            llm,
            interceptors=[
                LoggingInterceptor(
                    log_file="logs/custom_prompt.log",
                    log_to_console=True
                ),
                PromptTemplateInterceptor(prefix=system_prompt + "\n\n用户问题：\n")
            ]
        )
        
        response = wrapped_llm("什么是数据库索引？它有什么作用？")
        print(f"\n✅ 最终响应: {response}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


# ============================================================================
# 示例 3: 响应清理 - 移除不需要的标签
# ============================================================================

def example_response_cleaning():
    """清理模型输出中的特殊标签"""
    print("\n" + "="*80)
    print("📝 示例 3: 清理模型输出")
    print("="*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 创建响应清理拦截器
        wrapped_llm = LLMWrapper(
            llm,
            interceptors=[
                LoggingInterceptor(log_to_console=True),
                ResponseCleanerInterceptor(
                    strip_whitespace=True,
                    remove_tags=['<think>', '</think>', '<thought>', '</thought>'],
                    replace_patterns={
                        '...': '。',
                        '  ': ' '  # 移除多余空格
                    }
                )
            ]
        )
        
        response = wrapped_llm("解释一下什么是 JOIN 操作？")
        print(f"\n✅ 清理后的响应: {response}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


# ============================================================================
# 示例 4: 自定义拦截器 - 添加性能监控
# ============================================================================

class PerformanceInterceptor(LLMInterceptor):
    """性能监控拦截器 - 记录每次调用的耗时"""
    
    def __init__(self):
        self.start_time = None
        self.call_times = []
    
    def before_call(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """记录开始时间"""
        import time
        self.start_time = time.time()
        return prompt, kwargs
    
    def after_call(self, response: Any, **kwargs) -> Any:
        """计算耗时"""
        import time
        elapsed = time.time() - self.start_time
        self.call_times.append(elapsed)
        
        print(f"\n⏱️  本次调用耗时: {elapsed:.2f} 秒")
        if len(self.call_times) > 1:
            avg_time = sum(self.call_times) / len(self.call_times)
            print(f"📊 平均耗时: {avg_time:.2f} 秒 (共 {len(self.call_times)} 次调用)")
        
        return response
    
    def on_error(self, error: Exception, **kwargs) -> None:
        """记录错误时的耗时"""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"⚠️  调用失败，耗时: {elapsed:.2f} 秒")


def example_performance_monitoring():
    """添加性能监控"""
    print("\n" + "="*80)
    print("📝 示例 4: 性能监控")
    print("="*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 添加性能监控拦截器
        perf_interceptor = PerformanceInterceptor()
        
        wrapped_llm = LLMWrapper(
            llm,
            interceptors=[
                LoggingInterceptor(log_to_console=False, log_file="logs/perf_test.log"),
                perf_interceptor
            ]
        )
        
        # 执行多次调用
        questions = [
            "什么是主键？",
            "什么是外键？",
            "什么是索引？"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔍 问题 {i}: {question}")
            response = wrapped_llm(question)
            print(f"💬 回答: {response[:100]}..." if len(str(response)) > 100 else f"💬 回答: {response}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


# ============================================================================
# 示例 5: 高级用法 - 组合多个拦截器
# ============================================================================

class TokenCounterInterceptor(LLMInterceptor):
    """Token 计数拦截器"""
    
    def __init__(self):
        self.total_input_chars = 0
        self.total_output_chars = 0
    
    def before_call(self, prompt: str, **kwargs) -> tuple[str, dict]:
        self.total_input_chars += len(prompt)
        return prompt, kwargs
    
    def after_call(self, response: Any, **kwargs) -> Any:
        response_text = str(response)
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        
        self.total_output_chars += len(response_text)
        
        print(f"\n📈 Token 统计:")
        print(f"   输入字符数: {self.total_input_chars}")
        print(f"   输出字符数: {self.total_output_chars}")
        print(f"   总字符数: {self.total_input_chars + self.total_output_chars}")
        
        return response


def example_advanced_composition():
    """组合多个拦截器"""
    print("\n" + "="*80)
    print("📝 示例 5: 高级组合 - 日志 + 性能 + Token 统计 + 清理")
    print("="*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 创建多个拦截器
        logging_interceptor = LoggingInterceptor(
            log_file="logs/advanced_example.log",
            log_to_console=True,
            verbose=True
        )
        
        performance_interceptor = PerformanceInterceptor()
        token_counter_interceptor = TokenCounterInterceptor()
        
        prompt_interceptor = PromptTemplateInterceptor(
            prefix="请用简洁的语言回答：\n\n"
        )
        
        cleaner_interceptor = ResponseCleanerInterceptor(
            strip_whitespace=True,
            remove_tags=['<think>', '</think>']
        )
        
        # 组合所有拦截器
        # 执行顺序: logging -> performance -> token -> prompt (before_call)
        # 然后调用 LLM
        # 执行顺序: cleaner -> token -> performance -> logging (after_call)
        wrapped_llm = LLMWrapper(
            llm,
            interceptors=[
                logging_interceptor,
                performance_interceptor,
                token_counter_interceptor,
                prompt_interceptor,
                cleaner_interceptor
            ]
        )
        
        # 测试调用
        response = wrapped_llm("什么是数据库事务的 ACID 特性？")
        
        print(f"\n✅ 最终响应: {response}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


# ============================================================================
# 示例 6: 在 SQL Agent 中使用 LLM Wrapper
# ============================================================================

def example_with_sql_agent():
    """在 SQL Agent 中使用 LLM Wrapper"""
    print("\n" + "="*80)
    print("📝 示例 6: 在 SQL Agent 中使用")
    print("="*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from langchain_community.utilities import SQLDatabase
        from langchain_community.agent_toolkits import create_sql_agent
        from langchain.agents.agent_types import AgentType
        from ollama_model.model_config import OllamaModel, OllamaURI
        import os
        
        # 创建原始 LLM
        llm = OllamaLLM(
            model=OllamaModel,
            base_url=OllamaURI,
            temperature=0
        )
        
        # 包装 LLM
        wrapped_llm = create_custom_wrapper(
            llm,
            system_prompt="你是一个 SQL 专家。在生成 SQL 时要仔细考虑表结构。",
            remove_think_tags=True,
            log_interactions=True
        )
        
        # 连接数据库
        if os.path.exists("company.db"):
            db = SQLDatabase.from_uri("sqlite:///company.db")
            
            # 创建 SQL Agent（使用包装后的 LLM）
            agent = create_sql_agent(
                llm=wrapped_llm.llm,  # 注意：这里需要传入底层的 LLM
                db=db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # 执行查询
            result = agent.invoke({"input": "有多少个部门？"})
            print(f"\n✅ 查询结果: {result['output']}")
            print(f"📄 所有 LLM 交互已记录到日志文件")
        else:
            print("⚠️  未找到 company.db 数据库文件")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 示例 7: 使用工厂函数快速创建
# ============================================================================

def example_factory_functions():
    """使用便捷的工厂函数"""
    print("\n" + "="*80)
    print("📝 示例 7: 使用工厂函数")
    print("="*80)
    
    try:
        from langchain_ollama import OllamaLLM
        from ollama_model.model_config import OllamaModel, OllamaURI
        
        llm = OllamaLLM(model=OllamaModel, base_url=OllamaURI)
        
        # 方式1: 只记录日志
        logging_llm = create_logging_wrapper(llm)
        
        # 方式2: 完整的自定义配置
        custom_llm = create_custom_wrapper(
            llm,
            system_prompt="你是一个友好的助手。",
            remove_think_tags=True,
            log_interactions=True
        )
        
        print("✅ 创建了两个包装后的 LLM:")
        print("   1. logging_llm - 只记录日志")
        print("   2. custom_llm - 系统提示词 + 清理输出 + 日志")
        
        # 测试
        response = custom_llm("你好！")
        print(f"\n💬 响应: {response}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


# ============================================================================
# 主函数 - 运行所有示例
# ============================================================================

def main():
    """运行所有示例"""
    import os
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    print("\n" + "="*80)
    print("🎯 LLM Wrapper 完整使用示例")
    print("="*80)
    
    examples = [
        ("基础日志记录", example_basic_logging),
        ("自定义系统提示词", example_custom_prompt),
        ("清理模型输出", example_response_cleaning),
        ("性能监控", example_performance_monitoring),
        ("高级组合", example_advanced_composition),
        ("SQL Agent 集成", example_with_sql_agent),
        ("工厂函数", example_factory_functions),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "-"*80)
    choice = input("请选择要运行的示例 (输入数字，或 'all' 运行全部): ").strip()
    
    if choice.lower() == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"❌ 示例 '{name}' 执行失败: {e}")
                import traceback
                traceback.print_exc()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        try:
            func()
        except Exception as e:
            print(f"❌ 示例 '{name}' 执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ 无效的选择")
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)


if __name__ == '__main__':
    main()

