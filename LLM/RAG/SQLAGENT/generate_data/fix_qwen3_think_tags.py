"""
修复 Qwen3 生成的 SFT 数据中不成对的 <think> 标签
保留成对的标签（COT思考过程），只删除不成对的标签
"""
import json
import re
from typing import Tuple


def remove_unpaired_tags_with_stack(text: str) -> str:
    """
    使用栈匹配算法删除不成对的标签，保留成对的标签
    """
    # 找到所有 <think> 和 </think> 的位置
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
    
    # 按位置排序
    positions.sort(key=lambda x: x[1])
    
    # 使用栈匹配
    stack = []
    to_remove = []  # 需要删除的位置
    
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
    
    # 按位置倒序删除（避免位置偏移）
    to_remove.sort(reverse=True)
    
    result = text
    for start, end in to_remove:
        result = result[:start] + result[end:]
    
    return result


def check_and_fix_think_tags(text: str) -> Tuple[str, bool]:
    """
    检查并修复不成对的<think>标签
    保留成对的标签（COT思考过程），只删除不成对的标签
    """
    if not isinstance(text, str):
        return text, False
        
    # 计算 <think> 和 </think> 的数量
    open_tags = text.count('<think>')
    close_tags = text.count('</think>')
    
    has_issue = open_tags != close_tags
    
    if not has_issue:
        return text, False
    
    # 使用栈匹配方法
    fixed_text = remove_unpaired_tags_with_stack(text)
    
    # 清理多余的空白
    fixed_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', fixed_text)
    
    return fixed_text, True


def fix_qwen3_jsonl(input_file: str, output_file: str):
    """
    修复 Qwen3 生成的 JSONL 文件中的不成对 <think> 标签
    
    输入格式：每行一个 JSON，包含 call_id, type, prompts, generations 等字段
    输出格式：修复后的 JSONL
    """
    print(f"开始处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 80)
    
    issue_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            line = line.strip()
            
            if not line:
                continue
            
            try:
                # 解析 JSON
                data = json.loads(line)
                has_any_issue = False
                
                # 检查 prompts 字段（可能是列表）
                if 'prompts' in data and isinstance(data['prompts'], list):
                    for i, prompt in enumerate(data['prompts']):
                        if isinstance(prompt, str):
                            fixed_text, has_issue = check_and_fix_think_tags(prompt)
                            if has_issue:
                                if not has_any_issue:
                                    print(f"\n行 {line_num} (call_id: {data.get('call_id', 'N/A')}):")
                                    has_any_issue = True
                                print(f"  prompts[{i}] 存在不成对的标签")
                                data['prompts'][i] = fixed_text
                
                # 检查 generations 字段（可能是列表）
                if 'generations' in data and isinstance(data['generations'], list):
                    for i, gen in enumerate(data['generations']):
                        if isinstance(gen, str):
                            fixed_text, has_issue = check_and_fix_think_tags(gen)
                            if has_issue:
                                if not has_any_issue:
                                    print(f"\n行 {line_num} (call_id: {data.get('call_id', 'N/A')}):")
                                    has_any_issue = True
                                print(f"  generations[{i}] 存在不成对的标签")
                                data['generations'][i] = fixed_text
                
                if has_any_issue:
                    issue_count += 1
                
                # 写入修复后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"\n行 {line_num} JSON 解析错误: {e}")
                # 保持原样输出
                outfile.write(line + '\n')
            
            # 每处理1000行显示进度
            if line_num % 1000 == 0:
                print(f"已处理 {line_num} 行...")
    
    print("\n" + "=" * 80)
    print(f"处理完成!")
    print(f"总行数: {total_count}")
    print(f"存在问题的行数: {issue_count}")
    if total_count > 0:
        print(f"问题比例: {issue_count/total_count*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    # 修复 Qwen3 生成的数据
    input_file = "generate_data/sft_data_local_qwen3.jsonl"
    output_file = "generate_data/sft_data_local_qwen3_fixed.jsonl"
    
    fix_qwen3_jsonl(input_file, output_file)

