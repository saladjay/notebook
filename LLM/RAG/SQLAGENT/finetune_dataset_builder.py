"""
LLM 微调数据集生成器
用于收集和生成提高小模型对 LangChain Prompt 和数据库搜索能力的微调数据集

支持的微调格式：
1. Alpaca 格式（适用于大多数开源模型）
2. ShareGPT 格式（适用于 ChatGPT 类模型）
3. Completion 格式（用于传统续写任务）
4. Reasoning 格式（包含思考过程的推理数据）
"""
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import re
from config import get_config


class FinetuneDataCollector(BaseCallbackHandler):
    """
    收集 LLM 交互数据用于微调
    记录完整的交互流程，包括：
    - 原始 prompt（系统提示 + 用户问题）
    - 模型响应
    - ReAct 推理过程
    - SQL 查询和结果
    """
    
    def __init__(self):
        super().__init__()
        self.interactions = []
        self.current_interaction = None
        self.current_prompt = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """记录 LLM 输入"""
        if prompts:
            self.current_prompt = prompts[0]
            self.current_interaction = {
                'timestamp': datetime.now().isoformat(),
                'prompt': prompts[0],
                'response': None,
                'model': serialized.get('name', 'unknown')
            }
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """记录 LLM 输出"""
        if self.current_interaction and hasattr(response, 'generations'):
            for gen_list in response.generations:
                for gen in gen_list:
                    text = gen.text if hasattr(gen, 'text') else str(gen)
                    self.current_interaction['response'] = text
                    self.interactions.append(self.current_interaction.copy())
                    self.current_interaction = None
    
    def get_interactions(self) -> List[Dict[str, Any]]:
        """获取所有交互记录"""
        return self.interactions


class DatasetBuilder:
    """
    微调数据集构建器
    将收集的原始数据转换为各种微调格式
    """
    
    def __init__(self, data_source: List[Dict[str, Any]] = None):
        """
        初始化数据集构建器
        
        Args:
            data_source: 原始数据列表
        """
        self.raw_data = data_source or []
        self.processed_data = []
    
    def parse_react_response(self, response: str) -> Dict[str, Any]:
        """
        解析 ReAct 格式的响应
        提取 Thought、Action、Action Input、Observation、Final Answer
        """
        parsed = {
            'thoughts': [],
            'actions': [],
            'observations': [],
            'final_answer': None
        }
        
        # 提取 Thought
        thoughts = re.findall(r'Thought:(.*?)(?=Action:|Observation:|Final Answer:|$)', response, re.DOTALL)
        parsed['thoughts'] = [t.strip() for t in thoughts if t.strip()]
        
        # 提取 Action 和 Action Input
        actions = re.findall(r'Action:\s*(\w+)\s*Action Input:\s*(.*?)(?=Observation:|Thought:|$)', response, re.DOTALL)
        parsed['actions'] = [{'action': a[0].strip(), 'input': a[1].strip()} for a in actions]
        
        # 提取 Observation
        observations = re.findall(r'Observation:(.*?)(?=Thought:|Action:|Final Answer:|$)', response, re.DOTALL)
        parsed['observations'] = [o.strip() for o in observations if o.strip()]
        
        # 提取 Final Answer
        final_answer = re.search(r'Final Answer:(.*?)$', response, re.DOTALL)
        if final_answer:
            parsed['final_answer'] = final_answer.group(1).strip()
        
        return parsed
    
    def extract_sql_query(self, text: str) -> Optional[str]:
        """从文本中提取 SQL 查询"""
        # 查找 SELECT 语句
        sql_match = re.search(r'(SELECT.*?(?:;|$))', text, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        return None
    
    def extract_system_prompt(self, full_prompt: str) -> tuple[str, str]:
        """
        分离系统提示和用户问题
        
        Returns:
            (system_prompt, user_question)
        """
        # 查找 "Question:" 标记
        question_match = re.search(r'Question:\s*(.*?)(?:\nThought:|\n\n|$)', full_prompt, re.DOTALL)
        
        if question_match:
            question_start = question_match.start()
            system_prompt = full_prompt[:question_start].strip()
            user_question = question_match.group(1).strip()
        else:
            # 如果找不到，尝试其他分割方式
            parts = full_prompt.split('\n\n')
            if len(parts) >= 2:
                system_prompt = '\n\n'.join(parts[:-1])
                user_question = parts[-1]
            else:
                system_prompt = full_prompt
                user_question = ""
        
        return system_prompt, user_question
    
    # ============================================
    # 格式1: Alpaca 格式（最常用）
    # ============================================
    
    def to_alpaca_format(self, include_reasoning: bool = True) -> List[Dict[str, str]]:
        """
        转换为 Alpaca 格式
        
        格式：
        {
            "instruction": "系统指令",
            "input": "用户输入",
            "output": "模型输出"
        }
        
        Args:
            include_reasoning: 是否在输出中包含推理过程
        """
        alpaca_data = []
        
        for item in self.raw_data:
            prompt = item['prompt']
            response = item['response']
            
            if not response:
                continue
            
            # 分离系统提示和用户问题
            system_prompt, user_question = self.extract_system_prompt(prompt)
            
            # 解析响应
            parsed = self.parse_react_response(response)
            
            if include_reasoning:
                # 包含完整推理过程
                output = response
            else:
                # 只包含最终答案
                output = parsed['final_answer'] or response
            
            alpaca_data.append({
                "instruction": system_prompt,
                "input": user_question,
                "output": output
            })
        
        return alpaca_data
    
    # ============================================
    # 格式2: ShareGPT 格式（对话式）
    # ============================================
    
    def to_sharegpt_format(self) -> List[Dict[str, List[Dict[str, str]]]]:
        """
        转换为 ShareGPT 格式
        
        格式：
        {
            "conversations": [
                {"from": "system", "value": "系统提示"},
                {"from": "human", "value": "用户问题"},
                {"from": "gpt", "value": "AI回答"}
            ]
        }
        """
        sharegpt_data = []
        
        for item in self.raw_data:
            prompt = item['prompt']
            response = item['response']
            
            if not response:
                continue
            
            system_prompt, user_question = self.extract_system_prompt(prompt)
            
            sharegpt_data.append({
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": user_question},
                    {"from": "gpt", "value": response}
                ]
            })
        
        return sharegpt_data
    
    # ============================================
    # 格式3: SQL 专用格式（强化 SQL 生成能力）
    # ============================================
    
    def to_sql_focused_format(self) -> List[Dict[str, Any]]:
        """
        转换为 SQL 专用格式
        专注于提升模型的 SQL 生成能力
        
        格式：
        {
            "instruction": "Generate a SQL query to answer the question",
            "input": {
                "question": "用户问题",
                "schema": "数据库结构",
                "context": "额外上下文"
            },
            "output": {
                "sql": "SQL查询",
                "reasoning": "推理过程"
            }
        }
        """
        sql_data = []
        
        for item in self.raw_data:
            prompt = item['prompt']
            response = item['response']
            
            if not response:
                continue
            
            system_prompt, user_question = self.extract_system_prompt(prompt)
            
            # 提取数据库 schema（从 prompt 中）
            schema_match = re.search(r'CREATE TABLE.*?(?=\n\n|\nQuestion:)', prompt, re.DOTALL)
            schema = schema_match.group(0) if schema_match else ""
            
            # 提取 SQL 查询
            sql_query = self.extract_sql_query(response)
            
            # 解析推理过程
            parsed = self.parse_react_response(response)
            
            if sql_query:
                sql_data.append({
                    "instruction": "Generate a SQL query to answer the question based on the database schema.",
                    "input": {
                        "question": user_question,
                        "schema": schema,
                        "tables": re.findall(r'CREATE TABLE (\w+)', schema)
                    },
                    "output": {
                        "sql": sql_query,
                        "reasoning": parsed['thoughts'],
                        "final_answer": parsed['final_answer']
                    }
                })
        
        return sql_data
    
    # ============================================
    # 格式4: ReAct 推理格式（训练推理能力）
    # ============================================
    
    def to_react_reasoning_format(self) -> List[Dict[str, Any]]:
        """
        转换为 ReAct 推理格式
        专注于训练模型的推理和工具使用能力
        
        格式：
        {
            "instruction": "使用 ReAct 框架解决问题",
            "input": "问题描述 + 可用工具",
            "output": "完整的 ReAct 推理过程",
            "steps": [详细的推理步骤]
        }
        """
        react_data = []
        
        for item in self.raw_data:
            prompt = item['prompt']
            response = item['response']
            
            if not response:
                continue
            
            system_prompt, user_question = self.extract_system_prompt(prompt)
            
            # 提取工具定义
            tools_match = re.search(r'(sql_db_query.*?sql_db_query_checker.*?)(?=Use the following format:)', prompt, re.DOTALL)
            tools_desc = tools_match.group(1) if tools_match else ""
            
            # 解析推理步骤
            parsed = self.parse_react_response(response)
            
            # 构建步骤列表
            steps = []
            for i, (thought, action, obs) in enumerate(zip(
                parsed['thoughts'],
                parsed['actions'],
                parsed['observations'] + [None] * (len(parsed['thoughts']) - len(parsed['observations']))
            )):
                step = {
                    "step": i + 1,
                    "thought": thought
                }
                if action:
                    step["action"] = action
                if obs:
                    step["observation"] = obs
                steps.append(step)
            
            react_data.append({
                "instruction": "Solve the following database question using the ReAct (Reasoning + Acting) framework. Think step by step and use the provided tools.",
                "input": {
                    "question": user_question,
                    "available_tools": tools_desc,
                    "format_requirements": "Use the format: Thought -> Action -> Action Input -> Observation"
                },
                "output": response,
                "parsed_steps": steps,
                "final_answer": parsed['final_answer']
            })
        
        return react_data
    
    # ============================================
    # 格式5: Few-Shot 示例格式
    # ============================================
    
    def generate_few_shot_examples(self, max_examples: int = 10) -> List[Dict[str, str]]:
        """
        生成 Few-Shot 示例
        可用于在 Prompt 中添加示例
        
        格式：
        {
            "question": "用户问题",
            "sql": "SQL查询",
            "answer": "最终答案"
        }
        """
        examples = []
        
        for item in self.raw_data[:max_examples]:
            prompt = item['prompt']
            response = item['response']
            
            if not response:
                continue
            
            _, user_question = self.extract_system_prompt(prompt)
            sql_query = self.extract_sql_query(response)
            parsed = self.parse_react_response(response)
            
            if sql_query and parsed['final_answer']:
                examples.append({
                    "question": user_question,
                    "sql": sql_query,
                    "answer": parsed['final_answer']
                })
        
        return examples
    
    # ============================================
    # 数据质量控制
    # ============================================
    
    def filter_high_quality_data(self, min_response_length: int = 50) -> 'DatasetBuilder':
        """
        过滤高质量数据
        
        Args:
            min_response_length: 最小响应长度
        
        Returns:
            新的 DatasetBuilder 实例
        """
        filtered = []
        
        for item in self.raw_data:
            response = item.get('response', '')
            
            # 检查响应长度
            if len(response) < min_response_length:
                continue
            
            # 检查是否包含 Final Answer
            if 'Final Answer:' not in response:
                continue
            
            # 检查是否包含错误信息
            if 'error' in response.lower() and 'Final Answer:' not in response:
                continue
            
            filtered.append(item)
        
        return DatasetBuilder(filtered)
    
    def deduplicate(self) -> 'DatasetBuilder':
        """去重"""
        seen = set()
        unique = []
        
        for item in self.raw_data:
            # 使用 prompt 作为去重键
            key = item['prompt']
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        return DatasetBuilder(unique)
    
    # ============================================
    # 数据保存
    # ============================================
    
    def save(self, filename: str, format: str = 'alpaca', **kwargs):
        """
        保存数据集
        
        Args:
            filename: 输出文件名
            format: 数据格式 ('alpaca', 'sharegpt', 'sql', 'react', 'fewshot')
            **kwargs: 格式特定参数
        """
        if format == 'alpaca':
            data = self.to_alpaca_format(**kwargs)
        elif format == 'sharegpt':
            data = self.to_sharegpt_format()
        elif format == 'sql':
            data = self.to_sql_focused_format()
        elif format == 'react':
            data = self.to_react_reasoning_format()
        elif format == 'fewshot':
            data = self.generate_few_shot_examples(**kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存 {len(data)} 条数据到: {filename}")
        return len(data)
    
    def print_statistics(self):
        """打印数据统计"""
        print(f"\n{'='*80}")
        print("📊 数据集统计")
        print(f"{'='*80}")
        print(f"总数据量: {len(self.raw_data)}")
        
        if self.raw_data:
            avg_prompt_len = sum(len(item['prompt']) for item in self.raw_data) / len(self.raw_data)
            avg_response_len = sum(len(item.get('response', '')) for item in self.raw_data) / len(self.raw_data)
            
            print(f"平均 Prompt 长度: {avg_prompt_len:.0f} 字符")
            print(f"平均 Response 长度: {avg_response_len:.0f} 字符")
            
            # 统计包含 SQL 查询的数量
            sql_count = sum(1 for item in self.raw_data if self.extract_sql_query(item.get('response', '')))
            print(f"包含 SQL 查询: {sql_count} 条 ({sql_count/len(self.raw_data)*100:.1f}%)")
            
            # 统计包含 Final Answer 的数量
            final_answer_count = sum(1 for item in self.raw_data if 'Final Answer:' in item.get('response', ''))
            print(f"包含最终答案: {final_answer_count} 条 ({final_answer_count/len(self.raw_data)*100:.1f}%)")
        
        print(f"{'='*80}\n")


# ============================================
# 数据收集流程
# ============================================

def collect_training_data(questions: List[str], output_file: str = "raw_training_data.json"):
    """
    收集训练数据
    
    Args:
        questions: 问题列表
        output_file: 原始数据保存文件
    """
    print("="*80)
    print("🎯 开始收集微调训练数据")
    print("="*80)
    
    config = get_config()
    
    # 初始化
    llm = Ollama(
        model=config['ollama_model'],
        base_url=config.get('ollama_base_url'),
        temperature=0
    )
    
    db_uri = config['database_uri']
    if db_uri.startswith('sqlite://') and not db_uri.startswith('sqlite:///'):
        db_uri = 'sqlite:///' + db_uri[9:]
    db = SQLDatabase.from_uri(db_uri)
    
    print(f"✅ 已连接到数据库: {db.get_usable_table_names()}")
    
    # 创建数据收集器
    collector = FinetuneDataCollector()
    
    # 创建 Agent
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
    
    # 收集数据
    print(f"\n开始处理 {len(questions)} 个问题...")
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 处理: {question}")
        
        try:
            result = agent.invoke(
                {"input": question},
                config={"callbacks": [collector]}
            )
            print(f"  ✅ 完成: {result.get('output', '')[:100]}...")
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
    
    # 保存原始数据
    interactions = collector.get_interactions()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(interactions, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 已保存 {len(interactions)} 条原始数据到: {output_file}")
    
    return interactions


def process_and_generate_datasets(raw_data_file: str = "raw_training_data.json"):
    """
    处理原始数据并生成多种格式的数据集
    
    Args:
        raw_data_file: 原始数据文件
    """
    print("\n" + "="*80)
    print("🔄 处理数据并生成微调数据集")
    print("="*80)
    
    # 加载原始数据
    with open(raw_data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"加载了 {len(raw_data)} 条原始数据")
    
    # 创建数据集构建器
    builder = DatasetBuilder(raw_data)
    
    # 数据质量控制
    print("\n🔍 数据质量控制...")
    builder = builder.filter_high_quality_data().deduplicate()
    
    # 显示统计
    builder.print_statistics()
    
    # 生成各种格式
    print("\n📦 生成各种格式的数据集...")
    
    formats = [
        ('alpaca', 'finetune_alpaca.json', {'include_reasoning': True}),
        ('alpaca', 'finetune_alpaca_simple.json', {'include_reasoning': False}),
        ('sharegpt', 'finetune_sharegpt.json', {}),
        ('sql', 'finetune_sql_focused.json', {}),
        ('react', 'finetune_react_reasoning.json', {}),
        ('fewshot', 'fewshot_examples.json', {'max_examples': 10}),
    ]
    
    for fmt, filename, kwargs in formats:
        try:
            count = builder.save(filename, format=fmt, **kwargs)
            print(f"  ✅ {fmt.upper()}: {count} 条 -> {filename}")
        except Exception as e:
            print(f"  ❌ {fmt.upper()}: 失败 - {str(e)}")
    
    print("\n" + "="*80)
    print("✅ 数据集生成完成！")
    print("="*80)


# ============================================
# 主程序和示例
# ============================================

def main():
    """主函数 - 完整的数据收集和处理流程"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🎓 LLM 微调数据集生成器                                  ║
║     专注于 LangChain SQL Agent 能力提升                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 步骤1: 准备问题列表
    print("\n📝 步骤1: 准备训练问题")
    print("-"*80)
    
    # 从用户或文件加载问题
    questions = [
        # 基础查询
        "有多少个员工？",
        "有哪些部门？",
        "技术部有多少员工？",
        
        # 聚合查询
        "平均工资是多少？",
        "工资最高的员工是谁？",
        "每个部门的平均工资是多少？",
        
        # 复杂查询
        "工资超过8000的员工有哪些？",
        "技术部工资最高的前3名员工？",
        "哪个部门的员工最多？",
        
        # Join 查询
        "显示所有员工及其部门名称",
        "技术部的员工名单",
        
        # 可以添加更多问题...
    ]
    
    print(f"准备了 {len(questions)} 个训练问题")
    
    # 询问是否继续
    choice = input("\n是否开始收集数据？(y/n): ").strip().lower()
    if choice != 'y':
        print("已取消")
        return
    
    # 步骤2: 收集数据
    print(f"\n{'='*80}")
    print("📊 步骤2: 收集训练数据")
    print(f"{'='*80}")
    
    raw_data = collect_training_data(questions, "raw_training_data.json")
    
    # 步骤3: 处理和生成数据集
    process_and_generate_datasets("raw_training_data.json")
    
    # 步骤4: 使用建议
    print(f"\n{'='*80}")
    print("💡 使用建议")
    print(f"{'='*80}")
    print("""
已生成以下数据集文件：

1. finetune_alpaca.json (推荐)
   - 标准 Alpaca 格式，包含完整推理过程
   - 适用于: LLaMA, Qwen, Baichuan 等大多数开源模型
   - 微调命令示例:
     python train.py --data finetune_alpaca.json --model llama2-7b

2. finetune_alpaca_simple.json
   - 简化版 Alpaca 格式，只包含最终答案
   - 适用于: 需要快速响应的场景
   
3. finetune_sharegpt.json
   - ShareGPT 对话格式
   - 适用于: 对话式模型（ChatGLM, Qwen-Chat）
   
4. finetune_sql_focused.json
   - SQL 专用格式，强化 SQL 生成能力
   - 适用于: 专注于 SQL 生成的场景
   
5. finetune_react_reasoning.json
   - ReAct 推理格式，训练推理能力
   - 适用于: 需要强化推理和工具使用能力
   
6. fewshot_examples.json
   - Few-Shot 示例库
   - 用途: 在 Prompt 中添加示例，无需微调

微调建议：
- 数据量: 建议至少 100-1000 条高质量数据
- 模型选择: 7B-14B 参数的模型效果较好
- 训练参数: 
  * Learning Rate: 1e-5 到 5e-5
  * Batch Size: 4-8
  * Epochs: 3-5
  * LoRA Rank: 8-16 (如果使用 LoRA)

下一步：
1. 收集更多样化的问题（建议100+）
2. 使用生成的数据集进行微调
3. 评估微调后模型的效果
4. 根据效果调整数据集和训练参数
    """)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

