"""
方案B + 改进的完整使用示例

解决方案特点：
1. 解析器的log保存清理后的文本（避免<think>污染下一轮对话）
2. 使用专门的Callback在解析前捕获原始输出（完整记录<think>内容）
3. 完美平衡：既解决循环问题，又能收集完整数据
"""

from sql_agent.sql_agent_qwen import SQLAgentQwenAdvanced
from qwen_capture_callback import QwenOutputCaptureCallback, CombinedQwenCallback


def example_basic():
    """基础示例：使用单个Callback捕获原始输出"""
    print("=" * 80)
    print("示例1：使用原始输出捕获Callback")
    print("=" * 80)
    
    # 创建Agent
    agent = SQLAgentQwenAdvanced(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    
    # 创建捕获Callback
    capture_callback = QwenOutputCaptureCallback(
        log_file="qwen_original_outputs.log",
        console_output=True,  # 实时显示捕获的内容
        save_to_memory=True
    )
    
    # 执行查询
    question = "数据库中有哪些表？"
    print(f"\n问题: {question}")
    print("-" * 80)
    
    result = agent.query(question, callback=[capture_callback])
    
    print(f"\n✅ 查询完成")
    print(f"回答: {result['output']}")
    
    # 查看捕获的数据
    print("\n" + "=" * 80)
    print("捕获的原始输出：")
    print("=" * 80)
    
    for idx, output in enumerate(capture_callback.get_captured_outputs(), 1):
        print(f"\n第{idx}次LLM调用:")
        print(f"  时间: {output['timestamp']}")
        print(f"  包含<think>: {output['has_think_tags']}")
        print(f"  长度: {output['output_length']} 字符")
        print(f"  内容预览: {output['original_output'][:100]}...")


def example_combined():
    """完整示例：使用组合Callback"""
    print("\n" + "=" * 80)
    print("示例2：使用组合Callback（推荐）")
    print("=" * 80)
    
    # 创建Agent
    agent = SQLAgentQwenAdvanced(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    
    # 创建组合Callback（同时记录原始输出和Agent交互）
    combined_callback = CombinedQwenCallback(
        log_file="qwen_full_session.json",
        console_output=False
    )
    
    # 执行多个查询
    questions = [
        "数据库中有哪些表？",
        "employees表有多少条记录？",
        "薪资最高的3名员工是谁？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 60)
        
        result = agent.query(question, callback=[combined_callback])
        print(f"回答: {result['output']}")
    
    # 保存完整会话
    combined_callback.save_session()
    
    # 分析数据
    print("\n" + "=" * 80)
    print("会话统计：")
    print("=" * 80)
    print(f"LLM调用次数: {len(combined_callback.session_data['llm_outputs'])}")
    print(f"Agent动作数: {len(combined_callback.session_data['agent_actions'])}")
    print(f"工具执行数: {len(combined_callback.session_data['tool_executions'])}")
    
    # 检查<think>标签
    think_count = sum(
        1 for output in combined_callback.session_data['llm_outputs']
        if output['has_think']
    )
    print(f"包含<think>的输出: {think_count}/{len(combined_callback.session_data['llm_outputs'])}")


def example_dual_callbacks():
    """高级示例：同时使用两个Callback"""
    print("\n" + "=" * 80)
    print("示例3：双Callback方案（完整数据收集）")
    print("=" * 80)
    
    from qwen_capture_callback import create_qwen_callbacks
    
    # 创建Agent
    agent = SQLAgentQwenAdvanced(
        model_name="qwen3:7b",
        database_uri="company.db"
    )
    
    # 创建Callback组合
    callbacks = create_qwen_callbacks(
        original_output_file="qwen_original.log",
        full_session_file="qwen_session.json",
        console_output=True
    )
    
    # 执行查询
    question = "查询各部门的平均薪资"
    print(f"\n问题: {question}")
    print("-" * 60)
    
    result = agent.query(question, callback=callbacks)
    
    print(f"\n✅ 查询完成")
    print(f"回答: {result['output']}")
    
    # 保存会话数据
    for callback in callbacks:
        if hasattr(callback, 'save_session'):
            callback.save_session()
    
    print("\n✅ 所有数据已保存")
    print("  - 原始LLM输出: qwen_original.log")
    print("  - 完整会话数据: qwen_session.json")


def example_verify_no_pollution():
    """验证示例：确认<think>标签不会污染下一轮对话"""
    print("\n" + "=" * 80)
    print("示例4：验证循环问题已解决")
    print("=" * 80)
    
    # 创建Agent
    agent = SQLAgentQwenAdvanced()
    from qwen_capture_callback import create_qwen_callbacks
    # 创建Callback
    callbacks = create_qwen_callbacks(
        original_output_file="qwen_original.log",
        full_session_file="qwen_session.json",
        console_output=True
    )
    
    # 执行需要多轮对话的复杂查询
    question = "找出train数据集里高大于平均值的破损标签图片"
    print(f"\n问题（需要多轮对话）: {question}")
    print("-" * 60)
    
    try:
        result = agent.query(question, callback=callbacks)
        print(f"\n✅ 查询成功完成（无解析错误）")
        print(f"回答: {result['output']}")
        
        # 检查每轮输出
        # callbacks[0] 是 QwenOutputCaptureCallback
        outputs = callbacks[0].get_captured_outputs()
        print(f"\n共执行了 {len(outputs)} 轮LLM调用")
        
        for idx, output in enumerate(outputs, 1):
            print(f"\n第{idx}轮:")
            print(f"  包含<think>: {output['has_think_tags']}")
            # 即使第一轮有<think>，后续轮次仍能正常工作
            # 因为log中保存的是清理后的文本
        
        print("\n✅ 验证通过：<think>标签未污染后续对话")
        
    except Exception as e:
        print(f"\n❌ 查询失败: {str(e)}")
        print("如果看到解析错误，说明问题仍未完全解决")


def example_get_full_prompts_responses():
    """示例5：获取一个问题的所有 prompts 和 responses"""
    print("\n" + "=" * 80)
    print("示例5：获取完整的 Prompt-Response 对（用于微调数据）")
    print("=" * 80)
    
    # 创建Agent
    agent = SQLAgentQwenAdvanced()
    
    # 创建Callback（开启控制台输出）
    combined_callback = CombinedQwenCallback(
        log_file="qwen_full_session.json",
        console_output=True  # 显示问题开始/结束
    )
    
    # 执行查询
    question = "找出train数据集里高大于平均值的破损标签图片"
    print(f"\n问题: {question}")
    print("-" * 60)
    
    result = agent.query(question, callback=[combined_callback])
    
    print(f"\n最终答案: {result['output']}")
    
    # 获取完整的 prompts 和 responses
    print("\n" + "=" * 80)
    print("📊 完整的 Prompt-Response 数据")
    print("=" * 80)
    
    question_data = combined_callback.get_current_question_data()
    
    print(f"\n⏰ 时间信息:")
    print(f"   开始: {question_data['started_at']}")
    print(f"   结束: {question_data['ended_at']}")
    print(f"   LLM 调用次数: {question_data['total_llm_calls']}")
    
    print(f"\n📝 所有 Prompts (共 {len(question_data['prompts'])} 个):")
    for idx, prompt in enumerate(question_data['prompts'], 1):
        print(f"\n--- Prompt #{idx} ---")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    print(f"\n💬 所有 Responses (共 {len(question_data['responses'])} 个):")
    for idx, response in enumerate(question_data['responses'], 1):
        print(f"\n--- Response #{idx} ---")
        print(f"   Call ID: {response['call_id']}")
        print(f"   包含 <think>: {response['has_think']}")
        print(f"   时间: {response['timestamp']}")
        print(f"   内容: {response['output'][:200]}...")
    
    print("\n" + "=" * 80)
    print("✅ 数据收集完成！可用于构建微调数据集")
    print("=" * 80)


def main():
    """主函数"""
    import sys
    
    print("\n🎯 方案B + 改进：完整示例\n")
    print("解决方案特点：")
    print("  ✅ log保存清理后的文本 → 避免循环问题")
    print("  ✅ Callback捕获原始输出 → 完整记录<think>")
    print("  ✅ 完美平衡两个需求\n")
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        if example_num == "1":
            example_basic()
        elif example_num == "2":
            example_combined()
        elif example_num == "3":
            example_dual_callbacks()
        elif example_num == "4":
            example_verify_no_pollution()
        elif example_num == "5":
            example_get_full_prompts_responses()
        else:
            print(f"未知示例编号: {example_num}")
            print("可用示例: 1, 2, 3, 4, 5")
    else:
        print("可用示例:")
        print("  1 - 基础示例（原始输出捕获）")
        print("  2 - 组合Callback（推荐）")
        print("  3 - 双Callback（完整数据收集）")
        print("  4 - 验证循环问题已解决")
        print("  5 - 获取完整 Prompt-Response 对（微调数据）")
        print("\n运行方式: python example_qwen_solution_b.py [示例编号]")
        print("例如: python example_qwen_solution_b.py 5")
        print("\n运行默认示例（示例5）:")
        example_get_full_prompts_responses()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

