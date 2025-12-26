"""
SQL提取和比较的完整使用示例

演示：
1. 如何从Agent输出中提取SQL
2. 如何与人工标准答案比较
3. 如何批量验证SFT数据
"""
import json
from pathlib import Path
from sql_extractor_and_validator import (
    SQLExtractor,
    SQLComparator,
    SFTDataValidator,
    extract_sql_from_agent,
    compare_agent_sql_with_ground_truth
)
from config import get_config


# ============================================
# 示例1: 实时提取Agent执行的SQL
# ============================================

def example1_extract_sql_from_agent():
    """示例1: 实时从Agent中提取SQL"""
    print("\n" + "="*80)
    print("示例1: 从Agent中提取SQL")
    print("="*80)
    
    # 导入Agent
    from sql_agent.sql_agent_base import SQLAgentBase
    
    # 创建Agent
    agent_instance = SQLAgentBase()
    
    # 执行查询
    question = "有多少个员工？"
    print(f"\n问题: {question}")
    
    # 提取SQL
    sqls = extract_sql_from_agent(agent_instance.agent, question)
    
    print(f"\n提取到 {len(sqls)} 条SQL:")
    for i, sql in enumerate(sqls, 1):
        print(f"\n{i}. {sql}")
    
    return sqls


# ============================================
# 示例2: 与人工标准答案比较
# ============================================

def example2_compare_with_ground_truth():
    """示例2: 与标准答案比较"""
    print("\n" + "="*80)
    print("示例2: 与标准答案比较")
    print("="*80)
    
    from sql_agent.sql_agent_base import SQLAgentBase
    
    # 创建Agent
    agent_instance = SQLAgentBase()
    
    # 测试问题和标准答案
    question = "有多少个员工？"
    ground_truth_sql = "SELECT COUNT(*) FROM employees"
    
    print(f"\n问题: {question}")
    print(f"标准答案: {ground_truth_sql}")
    
    # 比较
    result = compare_agent_sql_with_ground_truth(
        agent_instance.agent,
        question,
        ground_truth_sql
    )
    
    print(f"\n提取的SQL: {result['final_sql']}")
    print(f"文本相似度: {result['comparison']['text_similarity']:.2%}")
    print(f"是否正确: {'✅' if result['comparison']['is_correct'] else '❌'}")
    
    if result['comparison']['results_comparison']:
        comp = result['comparison']['results_comparison']
        print(f"\n执行结果比较:")
        print(f"  生成的SQL结果: {comp['sql1_result']}")
        print(f"  标准答案结果: {comp['sql2_result']}")
        print(f"  结果是否匹配: {'✅' if comp['results_match'] else '❌'}")


# ============================================
# 示例3: 批量验证SFT数据
# ============================================

def example3_validate_sft_file():
    """示例3: 批量验证SFT文件"""
    print("\n" + "="*80)
    print("示例3: 批量验证SFT文件")
    print("="*80)
    
    # 假设的SFT文件路径
    sft_file = "finetune_alpaca.jsonl"
    
    if not Path(sft_file).exists():
        print(f"⚠️  示例文件不存在: {sft_file}")
        print("请先生成SFT数据")
        return
    
    # 创建验证器
    validator = SFTDataValidator()
    
    # 验证文件
    report = validator.validate_sft_file(sft_file)
    
    return report


# ============================================
# 示例4: 创建带标准答案的测试集
# ============================================

def example4_create_ground_truth_file():
    """示例4: 创建标准答案文件"""
    print("\n" + "="*80)
    print("示例4: 创建标准答案文件")
    print("="*80)
    
    # 定义问题和标准答案
    ground_truths = [
        {
            "question": "有多少个员工？",
            "sql": "SELECT COUNT(*) FROM employees"
        },
        {
            "question": "有哪些部门？",
            "sql": "SELECT DISTINCT department FROM employees"
        },
        {
            "question": "技术部有多少员工？",
            "sql": "SELECT COUNT(*) FROM employees WHERE department = '技术部'"
        },
        {
            "question": "工资最高的员工是谁？",
            "sql": "SELECT name FROM employees ORDER BY salary DESC LIMIT 1"
        },
        {
            "question": "平均工资是多少？",
            "sql": "SELECT AVG(salary) FROM employees"
        }
    ]
    
    # 保存到文件
    output_file = "ground_truth_sqls.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for gt in ground_truths:
            f.write(json.dumps(gt, ensure_ascii=False) + '\n')
    
    print(f"✅ 已创建标准答案文件: {output_file}")
    print(f"   包含 {len(ground_truths)} 条标准答案")
    
    return output_file


# ============================================
# 示例5: 使用标准答案验证SFT数据
# ============================================

def example5_validate_with_ground_truth():
    """示例5: 使用标准答案验证SFT数据"""
    print("\n" + "="*80)
    print("示例5: 使用标准答案验证SFT数据")
    print("="*80)
    
    sft_file = "finetune_alpaca.jsonl"
    gt_file = "ground_truth_sqls.jsonl"
    
    # 先创建标准答案文件
    if not Path(gt_file).exists():
        example4_create_ground_truth_file()
    
    if not Path(sft_file).exists():
        print(f"⚠️  SFT文件不存在: {sft_file}")
        return
    
    # 使用标准答案验证
    validator = SFTDataValidator()
    report = validator.validate_sft_file(sft_file, gt_file)
    
    return report


# ============================================
# 示例6: 单条数据验证（详细模式）
# ============================================

def example6_validate_single_item():
    """示例6: 验证单条数据（详细显示）"""
    print("\n" + "="*80)
    print("示例6: 验证单条数据（详细显示）")
    print("="*80)
    
    # 示例数据
    item = {
        "instruction": "根据数据库回答问题",
        "input": "有多少个员工？",
        "output": """我需要查询员工表

Action: sql_db_query
Action Input: SELECT COUNT(*) FROM employees
Observation: [(10,)]

Final Answer: 共有10名员工"""
    }
    
    ground_truth_sql = "SELECT COUNT(*) FROM employees"
    
    print(f"问题: {item['input']}")
    print(f"标准答案: {ground_truth_sql}")
    
    # 验证
    validator = SFTDataValidator()
    result = validator.validate_sft_item(item, ground_truth_sql)
    
    print(f"\n提取的SQL:")
    for i, sql in enumerate(result['extracted_sqls'], 1):
        print(f"  {i}. {sql}")
    
    print(f"\n验证结果: {'✅ 通过' if result['is_valid'] else '❌ 失败'}")
    
    if result['comparison_result']:
        comp = result['comparison_result']
        print(f"文本相似度: {comp['text_similarity']:.2%}")
        
        if comp['results_comparison']:
            rc = comp['results_comparison']
            print(f"\n执行结果:")
            print(f"  生成的SQL: {rc['sql1_result']}")
            print(f"  标准答案: {rc['sql2_result']}")
            print(f"  结果匹配: {'✅' if rc['results_match'] else '❌'}")
    
    if result['issues']:
        print(f"\n问题:")
        for issue in result['issues']:
            print(f"  - {issue}")


# ============================================
# 示例7: 批量测试Agent生成的SQL
# ============================================

def example7_batch_test_agent():
    """示例7: 批量测试Agent并与标准答案比较"""
    print("\n" + "="*80)
    print("示例7: 批量测试Agent")
    print("="*80)
    
    from sql_agent.sql_agent_base import SQLAgentBase
    
    # 创建Agent
    agent_instance = SQLAgentBase()
    
    # 测试用例
    test_cases = [
        {
            "question": "有多少个员工？",
            "expected_sql": "SELECT COUNT(*) FROM employees"
        },
        {
            "question": "有哪些部门？",
            "expected_sql": "SELECT DISTINCT department FROM employees"
        }
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}: {case['question']}")
        
        result = compare_agent_sql_with_ground_truth(
            agent_instance.agent,
            case['question'],
            case['expected_sql']
        )
        
        results.append(result)
        
        is_correct = result['comparison']['is_correct']
        print(f"  结果: {'✅ 正确' if is_correct else '❌ 错误'}")
        if not is_correct:
            print(f"  生成的SQL: {result['final_sql']}")
            print(f"  期望的SQL: {case['expected_sql']}")
    
    # 统计
    total = len(results)
    correct = sum(1 for r in results if r['comparison']['is_correct'])
    
    print(f"\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"总测试数: {total}")
    print(f"正确数: {correct}")
    print(f"正确率: {correct/total*100:.1f}%")
    
    return results


# ============================================
# 主函数
# ============================================

def main():
    """主函数 - 运行所有示例"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🎯 SQL提取和比较完整示例                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n选择要运行的示例:")
    print("1. 从Agent中提取SQL")
    print("2. 与标准答案比较")
    print("3. 批量验证SFT文件")
    print("4. 创建标准答案文件")
    print("5. 使用标准答案验证SFT数据")
    print("6. 验证单条数据（详细）")
    print("7. 批量测试Agent")
    print("8. 运行所有示例")
    
    choice = input("\n请选择 (1-8): ").strip()
    
    if choice == '1':
        example1_extract_sql_from_agent()
    elif choice == '2':
        example2_compare_with_ground_truth()
    elif choice == '3':
        example3_validate_sft_file()
    elif choice == '4':
        example4_create_ground_truth_file()
    elif choice == '5':
        example5_validate_with_ground_truth()
    elif choice == '6':
        example6_validate_single_item()
    elif choice == '7':
        example7_batch_test_agent()
    elif choice == '8':
        # 运行所有示例
        try:
            example6_validate_single_item()  # 先运行不需要外部文件的
            example4_create_ground_truth_file()
            # 其他示例需要Agent运行，可能比较慢
            print("\n提示: 其他示例需要Agent运行，可能需要一些时间")
        except Exception as e:
            print(f"运行示例时出错: {str(e)}")
    else:
        print("无效的选择")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


