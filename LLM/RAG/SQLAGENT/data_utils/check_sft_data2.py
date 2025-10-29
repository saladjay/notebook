import json
import re
from typing import Tuple


def check_and_fix_think_tags(text: str) -> Tuple[str, bool]:
    """
    检查并修复不成对的<think>标签
    保留成对的标签（COT思考过程），只删除不成对的标签
    
    Args:
        text: 要检查的文本
        
    Returns:
        (fixed_text, has_issue): 修复后的文本和是否存在问题
    """
    # 计算 <think> 和 </think> 的数量
    open_tags = text.count('<think>')
    close_tags = text.count('</think>')
    
    has_issue = open_tags != close_tags
    
    if not has_issue:
        return text, False
    
    print(f"  发现不成对的标签: <think>={open_tags}, </think>={close_tags}")
    
    # 策略：保留成对的标签，删除多余的不成对标签
    fixed_text = text
    
    if open_tags > close_tags:
        # 有多余的开始标签，需要删除多余的 <think>
        # 方法：找到所有成对的，标记它们，删除未配对的 <think>
        excess = open_tags - close_tags
        
        # 从后往前查找并删除多余的 <think>（没有对应 </think> 的）
        for _ in range(excess):
            # 找到最后一个没有配对的 <think>
            # 简单策略：找到最后一个 <think>，如果它后面没有 </think>，就删除它
            last_open = fixed_text.rfind('<think>')
            if last_open != -1:
                # 检查这个 <think> 后面是否有 </think>
                after_text = fixed_text[last_open:]
                if '</think>' not in after_text:
                    # 这个 <think> 没有配对，删除它
                    fixed_text = fixed_text[:last_open] + fixed_text[last_open + len('<think>'):]
                else:
                    # 这个 <think> 有配对，需要找前面没有配对的
                    # 采用更复杂的策略：模拟栈匹配
                    break
        
        # 如果上面的简单策略不够，使用栈匹配方法
        if fixed_text.count('<think>') > fixed_text.count('</think>'):
            fixed_text = remove_unpaired_tags_with_stack(fixed_text)
            
    elif close_tags > open_tags:
        # 有多余的结束标签，需要删除多余的 </think>
        excess = close_tags - open_tags
        
        # 从前往后查找并删除多余的 </think>（没有对应 <think> 的）
        for _ in range(excess):
            # 找到第一个没有配对的 </think>
            first_close = fixed_text.find('</think>')
            if first_close != -1:
                # 检查这个 </think> 前面是否有 <think>
                before_text = fixed_text[:first_close]
                if '<think>' not in before_text:
                    # 这个 </think> 没有配对，删除它
                    fixed_text = fixed_text[:first_close] + fixed_text[first_close + len('</think>'):]
                else:
                    # 需要更复杂的匹配
                    break
        
        # 如果还有多余的，使用栈匹配方法
        if fixed_text.count('</think>') > fixed_text.count('<think>'):
            fixed_text = remove_unpaired_tags_with_stack(fixed_text)
    
    # 清理多余的空白
    fixed_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', fixed_text)
    
    return fixed_text, True


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


def process_jsonl_file(input_file: str, output_file: str):
    """
    处理 JSONL 文件，修复不成对的 <think> 标签
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
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
            
            try:
                # 解析 JSON
                data = json.loads(line.strip())
                
                # 检查并修复每个字段
                fields_to_check = ['instruction', 'input', 'output']
                has_any_issue = False
                
                for field in fields_to_check:
                    if field in data and isinstance(data[field], str):
                        fixed_text, has_issue = check_and_fix_think_tags(data[field])
                        if has_issue:
                            if not has_any_issue:
                                print(f"\n行 {line_num}:")
                                has_any_issue = True
                            print(f"  字段 '{field}' 存在不成对的标签")
                            data[field] = fixed_text
                
                if has_any_issue:
                    issue_count += 1
                
                # 写入修复后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"\n行 {line_num} JSON 解析错误: {e}")
                # 保持原样输出
                outfile.write(line)
            
            # 每处理1000行显示进度
            if line_num % 1000 == 0:
                print(f"已处理 {line_num} 行...")
    
    print("\n" + "=" * 80)
    print(f"处理完成!")
    print(f"总行数: {total_count}")
    print(f"存在问题的行数: {issue_count}")
    print(f"问题比例: {issue_count/total_count*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    # 处理 finetune_alpaca2.jsonl
    input_file = "finetune_alpaca2_fixed2.jsonl"
    output_file = "finetune_alpaca2_fixed3.jsonl"
    
    process_jsonl_file(input_file, output_file)
