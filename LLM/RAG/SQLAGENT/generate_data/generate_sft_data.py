import os
import json
import logging

from sql_agent.sql_agent_demo import SQLAgentDemo
from sql_agent.sql_agent_qwen import SQLAgentQwen3
from callbacks import LLMLoggingCallback

# 关闭 httpx 的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
# 如果还有其他干扰日志，也可以关闭
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

callbacks = [LLMLoggingCallback(json_file="generate_data\sft_data7_20251103.json", log_to_file=False, log_to_console=False, save_json=True)]

agent = SQLAgentQwen3()

def generate_sft_data():
    with open(r"generate_data\ai_generate_query.txt", 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        for i, line in enumerate(lines):
            print(f"Generating data for question {i+1}: {line}")
            if i == 1:
                exit()
            # if i < 64:
            #     continue
            # if i > 150:
            #     break
            subject = line.split(":")[0]
            question = ":".join(line.split(":")[1:])
            question = "2028_03_07数据集中每个标签类别分别有种Tag"
            result = agent.query(question, callbacks)


