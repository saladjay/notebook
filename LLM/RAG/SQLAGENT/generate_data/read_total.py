import os
import json


def read_json_lines(file_path, num_lines=10):
    """
    读取大JSON文件的前几行（适用于JSON Lines格式，每行一个JSON对象）
    
    Args:
        file_path: JSON文件路径
        num_lines: 要读取的行数，默认10行
    """
    print(f"读取文件: {file_path}")
    print(f"前 {num_lines} 行内容:\n")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                
                # 尝试解析JSON并美化打印
                try:
                    json_obj = json.loads(line.strip())
                    print(f"\n第 {i+1} 行:")
                    print(json.dumps(json_obj, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    # 如果不是有效JSON，直接打印原始文本
                    print(f"\n第 {i+1} 行 (原始文本):")
                    print(line.strip())
                
                print("-" * 80)
        
        print(f"\n✓ 已成功读取前 {num_lines} 行")
    
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {file_path}")
    except Exception as e:
        print(f"错误: {e}")


def read_file_lines(file_path, num_lines=10):
    """
    简单读取文件的前几行（不解析JSON，直接打印原始文本）
    
    Args:
        file_path: 文件路径
        num_lines: 要读取的行数，默认10行
    """
    print(f"读取文件: {file_path}")
    print(f"前 {num_lines} 行原始内容:\n")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(f"{i+1}: {line.rstrip()}")
        
        print("=" * 80)
        print(f"\n✓ 已成功读取前 {num_lines} 行")
    
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {file_path}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    # 使用示例
    
    # 方法1: 读取JSON Lines格式（每行一个JSON对象）
    json_file = "sft_data_local_qwen3.json"
    if os.path.exists(json_file):
        read_json_lines(json_file, num_lines=30)
    else:
        print(f"文件不存在: {json_file}")
        print(f"当前目录: {os.getcwd()}")
    
    print("\n" + "=" * 80 + "\n")
    
    # 方法2: 简单读取原始文本行
    # read_file_lines(json_file, num_lines=10)
    
    # 如果需要读取其他文件，修改文件路径即可
    # read_json_lines("path/to/your/large_file.json", num_lines=20)