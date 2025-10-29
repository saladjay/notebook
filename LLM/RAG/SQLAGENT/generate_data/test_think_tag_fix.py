"""
测试 think 标签修复功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generate_data.generate_sft_data_local_qwen3 import remove_unpaired_think_tags, clean_data_think_tags


def test_remove_unpaired_think_tags():
    print("=" * 80)
    print("测试 remove_unpaired_think_tags 函数")
    print("=" * 80)
    
    # 测试用例 1: 成对的标签 - 应该保留
    test1 = "<think>这是思考过程</think>最终答案"
    result1 = remove_unpaired_think_tags(test1)
    print(f"\n✅ 测试1 - 成对标签（应保留）:")
    print(f"输入: {test1}")
    print(f"输出: {result1}")
    assert result1 == test1, "成对标签应该保留"
    
    # 测试用例 2: 多个成对的标签 - 应该保留
    test2 = "<think>思考1</think>答案1\n<think>思考2</think>答案2"
    result2 = remove_unpaired_think_tags(test2)
    print(f"\n✅ 测试2 - 多个成对标签（应保留）:")
    print(f"输入: {test2}")
    print(f"输出: {result2}")
    assert result2 == test2, "多个成对标签应该保留"
    
    # 测试用例 3: 只有开始标签 - 应该删除
    test3 = "<think>这是思考过程，但没有结束标签"
    result3 = remove_unpaired_think_tags(test3)
    print(f"\n✅ 测试3 - 只有开始标签（应删除）:")
    print(f"输入: {test3}")
    print(f"输出: {result3}")
    assert '<think>' not in result3, "不成对的开始标签应该删除"
    
    # 测试用例 4: 只有结束标签 - 应该删除
    test4 = "没有开始标签</think>后面的内容"
    result4 = remove_unpaired_think_tags(test4)
    print(f"\n✅ 测试4 - 只有结束标签（应删除）:")
    print(f"输入: {test4}")
    print(f"输出: {result4}")
    assert '</think>' not in result4, "不成对的结束标签应该删除"
    
    # 测试用例 5: 混合情况 - 保留成对，删除不成对
    test5 = "<think>成对思考</think>答案<think>不成对思考"
    result5 = remove_unpaired_think_tags(test5)
    print(f"\n✅ 测试5 - 混合情况:")
    print(f"输入: {test5}")
    print(f"输出: {result5}")
    assert result5.count('<think>') == result5.count('</think>'), "应该只保留成对标签"
    assert '<think>成对思考</think>' in result5, "成对标签应该保留"
    
    # 测试用例 6: 复杂嵌套 - <think>内容1</think> 文本 <think>内容2
    test6 = "<think>内容1</think>中间文本<think>内容2"
    result6 = remove_unpaired_think_tags(test6)
    print(f"\n✅ 测试6 - 复杂混合:")
    print(f"输入: {test6}")
    print(f"输出: {result6}")
    assert '<think>内容1</think>' in result6, "成对标签应该保留"
    
    # 测试用例 7: 数量相等但顺序错误
    test7 = "</think>错误的顺序<think>"
    result7 = remove_unpaired_think_tags(test7)
    print(f"\n✅ 测试7 - 数量相等但顺序错误:")
    print(f"输入: {test7}")
    print(f"输出: {result7}")
    assert '<think>' not in result7 and '</think>' not in result7, "顺序错误的标签应该全部删除"
    
    print(f"\n{'='*80}")
    print("✅ 所有测试通过!")
    print(f"{'='*80}\n")


def test_clean_data_think_tags():
    print("=" * 80)
    print("测试 clean_data_think_tags 函数")
    print("=" * 80)
    
    # 测试数据结构
    test_data = {
        'question': '测试问题',
        'status': 'success',
        'responses': [
            {
                'input': '<think>思考1</think>输入内容<think>不成对',
                'output': '<think>成对思考</think>答案内容'
            },
            {
                'input': '正常输入',
                'output': '</think>只有结束标签的输出'
            }
        ]
    }
    
    print(f"\n原始数据:")
    print(f"{test_data}")
    
    cleaned_data = clean_data_think_tags(test_data)
    
    print(f"\n清理后数据:")
    print(f"{cleaned_data}")
    
    # 验证清理结果
    assert '<think>成对思考</think>' in cleaned_data['responses'][0]['output'], "成对标签应该保留"
    assert cleaned_data['responses'][0]['input'].count('<think>') == cleaned_data['responses'][0]['input'].count('</think>'), "不成对标签应该被删除"
    assert '</think>' not in cleaned_data['responses'][1]['output'] or '<think>' in cleaned_data['responses'][1]['output'], "单独的结束标签应该被删除"
    
    print(f"\n{'='*80}")
    print("✅ 数据清理测试通过!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_remove_unpaired_think_tags()
    test_clean_data_think_tags()
    print("\n🎉 所有测试完成!")

