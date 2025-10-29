"""
使用 Qwen3 模型生成 SFT 训练数据

🔧 不成对 <think> 标签的修复机制：
通过 LangChain 的 Callback 机制在源头修复，而不是事后清理数据
- CombinedQwenCallback.on_llm_end() 会在 LLM 输出时立即清理不成对的标签
- QwenReActOutputParser 在解析时也会处理标签
这样确保整个数据流都是干净的，符合 LangChain 的设计理念
"""
import os
import json

from sql_agent.sql_agent_demo import SQLAgentDemo
from callbacks import LLMLoggingCallback
from qwen_capture_callback import CombinedQwenCallback
from sql_agent.sql_agent_qwen import SQLAgentQwen3

# 🔧 使用修复后的 CombinedQwenCallback，自动清理不成对的 <think> 标签
callback1 = CombinedQwenCallback(log_file="generate_data\sft_data_local_qwen3_2.json", console_output=False)
# callbacks = []
agent = SQLAgentDemo()

def generate_sft_data():
    with open(r"generate_data\ai_generate_query.txt", 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        for i, line in enumerate(lines):
            subject, question = line.split(":")
            try:
                result = agent.query(question, [callback1])
                data = callback1.get_current_question_data()
                data['status'] = 'success'
                data['question'] = question
            except Exception as e:
                # ✅ 即使出错也能获取已捕获的数据
                print(f"\n❌ 问题 {i+1} 失败: {question}")
                print(f"错误: {str(e)}")
                
                # 获取错误时的数据
                data = callback1.get_current_question_data()
                data['status'] = 'error'
                data['error'] = str(e)
                data['question'] = question
                
                # 打印最后一次 response（导致错误的输出）
                if data['responses']:
                    last_response = data['responses'][-1]
                    print(f"\n🔍 导致错误的 Response:")
                    print(f"{'='*80}")
                    print(last_response['output'])
                    print(f"{'='*80}\n")
            
            # ✅ 无论成功或失败都保存数据
            # 注意：不需要在这里清理 <think> 标签，已经通过 LangChain 机制在源头修复
            with open(r"generate_data\sft_data_local_qwen3_log.json", 'a', encoding='utf-8') as f:
                content = json.dumps(data, ensure_ascii=False)
                print(content, file=f)


