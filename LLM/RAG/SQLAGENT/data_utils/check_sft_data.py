import json
import os
import argparse

# 检查是否有不成对的</think>和<think>标签
def check_think_tags(content):
    def small_loop(content1, content2):
        think_start = content1.find(r'<think>')
        think_end = content2.find(r'</think>')
        if think_start == -1 and think_end == -1:
            return True
        if think_start != -1 or think_end == -1:
            print("has unmatched </think>")
            return False
        if think_start == -1 or think_end != -1:
            print("has unmatched <think>")
            return False
        if think_start > think_end:
            return False
        return think_start, think_end
    content1 = content
    content2 = content
    while True:
        result = small_loop(content1, content2)
        if result is True:
            return True
        if result is False:
            return False
        content1 = content1[result[0]+1:]
        content2 = content2[result[1]+1:]
    return True



def check_sft_data(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            # if i > 15:
            #     exit()
            line = line.strip()
            content = json.loads(line)
            instruction = content["instruction"]
            output = content["output"]
            input = content["input"]
            if not check_think_tags(instruction):
                print(f"{input_file} has invalid think tags at line {i} instruction")
                # print(line)
                continue
            if not check_think_tags(output):
                print(f"{input_file} has invalid think tags at line {i} output")
                # print(line)
                continue
            if not check_think_tags(input):
                print(f"{input_file} has invalid think tags at line {i} input")
                # print(line)
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    check_sft_data(args.input_file)