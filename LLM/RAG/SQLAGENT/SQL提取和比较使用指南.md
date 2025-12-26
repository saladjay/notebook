# SQL提取和比较使用指南

## 📌 问题场景

在制作SFT训练数据时，你需要：
1. **提取Agent最后执行的SQL** - 从Agent的输出中准确提取SQL语句
2. **与人工标准答案比较** - 验证生成的SQL是否与预期一致
3. **验证SQL正确性** - 确保SQL可以正确执行

---

## 🎯 核心功能

我已经为你创建了完整的工具链，包含以下核心类：

### 1. **SQLExtractor** - SQL提取器

从各种来源提取SQL语句：

```python
from sql_extractor_and_validator import SQLExtractor

extractor = SQLExtractor()

# 从文本中提取
sqls = extractor.extract_from_text("Action Input: SELECT * FROM employees")

# 从Agent输出中提取
sqls = extractor.extract_from_agent_output(agent_result)

# 从回调历史中提取
sqls = extractor.extract_from_callback_history(callback)
```

### 2. **SQLComparator** - SQL比较器

比较生成的SQL与标准答案：

```python
from sql_extractor_and_validator import SQLComparator

comparator = SQLComparator()

# 完整比较（文本相似度 + 执行结果）
result = comparator.compare(
    generated_sql="SELECT * FROM employees",
    ground_truth_sql="SELECT * FROM employees"
)

print(f"是否正确: {result['is_correct']}")
print(f"文本相似度: {result['text_similarity']:.2%}")
```

### 3. **SFTDataValidator** - SFT数据验证器

批量验证SFT训练数据：

```python
from sql_extractor_and_validator import SFTDataValidator

validator = SFTDataValidator()

# 验证整个文件
report = validator.validate_sft_file(
    "finetune_alpaca.jsonl",
    ground_truth_file="ground_truth_sqls.jsonl"  # 可选
)
```

---

## 🚀 快速开始

### 方式1: 实时提取Agent执行的SQL

```python
from sql_extractor_and_validator import extract_sql_from_agent
from sql_agent.sql_agent_base import SQLAgentBase

# 创建Agent
agent_instance = SQLAgentBase()

# 执行查询并提取SQL
question = "有多少个员工？"
sqls = extract_sql_from_agent(agent_instance.agent, question)

print(f"提取到的SQL: {sqls}")
# 输出: ['SELECT COUNT(*) FROM employees']
```

### 方式2: 与标准答案比较

```python
from sql_extractor_and_validator import compare_agent_sql_with_ground_truth
from sql_agent.sql_agent_base import SQLAgentBase

# 创建Agent
agent_instance = SQLAgentBase()

# 执行查询并与标准答案比较
result = compare_agent_sql_with_ground_truth(
    agent_instance.agent,
    question="有多少个员工？",
    ground_truth_sql="SELECT COUNT(*) FROM employees"
)

print(f"是否正确: {result['comparison']['is_correct']}")
print(f"生成的SQL: {result['final_sql']}")
print(f"文本相似度: {result['comparison']['text_similarity']:.2%}")
```

### 方式3: 批量验证SFT文件

```bash
# 运行示例脚本
python example_sql_extraction_and_comparison.py
```

---

## 📖 完整使用流程

### 步骤1: 准备标准答案文件

创建 `ground_truth_sqls.jsonl`，每行一个JSON对象：

```json
{"question": "有多少个员工？", "sql": "SELECT COUNT(*) FROM employees"}
{"question": "有哪些部门？", "sql": "SELECT DISTINCT department FROM employees"}
{"question": "技术部有多少员工？", "sql": "SELECT COUNT(*) FROM employees WHERE department = '技术部'"}
```

**快速生成方法**：

```python
from example_sql_extraction_and_comparison import example4_create_ground_truth_file

example4_create_ground_truth_file()
```

### 步骤2: 生成SFT数据

使用你现有的方法生成SFT数据：

```python
from sql_agent.sql_agent_qwen import SQLAgentQwen3
from callbacks import LLMLoggingCallback

# 生成数据...
```

### 步骤3: 提取和验证SQL

#### 方法A: 实时验证（推荐）

在生成数据时就进行验证：

```python
from sql_extractor_and_validator import SQLComparator, SQLExtractor

comparator = SQLComparator()
extractor = SQLExtractor()

# 标准答案字典
ground_truths = {
    "有多少个员工？": "SELECT COUNT(*) FROM employees",
    "有哪些部门？": "SELECT DISTINCT department FROM employees"
}

for question in questions:
    # 生成回答
    result = agent.query(question)
    
    # 提取SQL
    sqls = extractor.extract_from_text(result)
    
    if sqls:
        final_sql = sqls[-1]  # 取最后一个
        
        # 与标准答案比较
        if question in ground_truths:
            comparison = comparator.compare(final_sql, ground_truths[question])
            
            if comparison['is_correct']:
                print(f"✅ {question}: SQL正确")
                save_to_dataset(result)
            else:
                print(f"❌ {question}: SQL错误")
                print(f"   生成: {final_sql}")
                print(f"   期望: {ground_truths[question]}")
```

#### 方法B: 批量验证

生成完所有数据后批量验证：

```python
from sql_extractor_and_validator import SFTDataValidator

validator = SFTDataValidator()

# 验证SFT文件，同时比对标准答案
report = validator.validate_sft_file(
    "finetune_alpaca.jsonl",
    ground_truth_file="ground_truth_sqls.jsonl"
)

print(f"有效数据: {report['valid_items']}/{report['total_items']}")
print(f"与标准答案一致: {report['correct_with_gt']}/{report['with_ground_truth']}")
```

### 步骤4: 分析和改进

根据验证结果改进数据：

```python
# 查看验证详情
for detail in report['validation_details'][:5]:
    print(f"\n问题: {detail['question']}")
    print(f"错误: {detail['validation']['issues']}")
    
    if detail['validation']['comparison_result']:
        comp = detail['validation']['comparison_result']
        print(f"生成的SQL: {comp['generated_sql']}")
        print(f"标准答案: {comp['ground_truth_sql']}")
        print(f"文本相似度: {comp['text_similarity']:.2%}")
```

---

## 📊 验证报告解读

运行验证后会生成详细报告：

```
================================================================================
📊 验证报告
================================================================================
文件: finetune_alpaca.jsonl
总数据量: 150

📈 SQL提取:
  包含SQL: 145 (96.7%)

✅ SQL有效性:
  有效SQL: 130 (89.7%)
  无效SQL: 15

🎯 与标准答案比较:
  有标准答案: 100
  与标准答案一致: 85 (85.0%)
```

### 关键指标说明

| 指标 | 说明 | 良好阈值 |
|------|------|----------|
| SQL提取率 | 能从输出中提取到SQL的比例 | ≥ 95% |
| SQL有效性 | SQL能正确执行的比例 | ≥ 90% |
| 标准答案一致率 | 与ground truth一致的比例 | ≥ 85% |

---

## 💡 高级用法

### 1. 自定义SQL提取规则

如果默认提取规则不够用，可以扩展：

```python
from sql_extractor_and_validator import SQLExtractor

class MyCustomExtractor(SQLExtractor):
    @staticmethod
    def extract_from_text(text: str):
        # 先使用父类方法
        sqls = SQLExtractor.extract_from_text(text)
        
        # 添加自定义规则
        # 例如：提取注释中的SQL
        custom_sqls = re.findall(r'-- SQL: (.*?)$', text, re.MULTILINE)
        sqls.extend(custom_sqls)
        
        return sqls
```

### 2. 自定义比较逻辑

```python
from sql_extractor_and_validator import SQLComparator

class MyCustomComparator(SQLComparator):
    def compare(self, sql1, sql2, **kwargs):
        # 先使用父类方法
        result = super().compare(sql1, sql2, **kwargs)
        
        # 添加自定义逻辑
        # 例如：检查SQL是否使用了索引
        if 'JOIN' in sql1.upper() and 'ON' not in sql1.upper():
            result['issues'].append('JOIN语句缺少ON条件')
            result['is_correct'] = False
        
        return result
```

### 3. 语义等价性检查

有时SQL写法不同但结果相同：

```python
# 这两个SQL语义等价
sql1 = "SELECT COUNT(*) FROM employees WHERE department = '技术部'"
sql2 = "SELECT COUNT(*) FROM employees WHERE department = '技术部' AND 1=1"

comparator = SQLComparator()
result = comparator.compare(sql1, sql2, check_results=True)

# 文本相似度可能不高，但执行结果相同
print(f"文本相似度: {result['text_similarity']:.2%}")  # 可能较低
print(f"结果匹配: {result['results_comparison']['results_match']}")  # True
print(f"是否正确: {result['is_correct']}")  # True
```

---

## 🛠️ 实用工具函数

### 工具1: 批量测试Agent

```python
def batch_test_agent(agent, test_cases):
    """
    批量测试Agent并统计准确率
    
    Args:
        agent: SQL Agent实例
        test_cases: 测试用例列表 [{"question": "...", "expected_sql": "..."}]
    
    Returns:
        测试报告
    """
    from sql_extractor_and_validator import compare_agent_sql_with_ground_truth
    
    results = []
    for case in test_cases:
        result = compare_agent_sql_with_ground_truth(
            agent,
            case['question'],
            case['expected_sql']
        )
        results.append(result)
    
    # 统计
    total = len(results)
    correct = sum(1 for r in results if r['comparison']['is_correct'])
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'details': results
    }

# 使用
from sql_agent.sql_agent_base import SQLAgentBase

test_cases = [
    {"question": "有多少个员工？", "expected_sql": "SELECT COUNT(*) FROM employees"},
    {"question": "有哪些部门？", "expected_sql": "SELECT DISTINCT department FROM employees"}
]

agent_instance = SQLAgentBase()
report = batch_test_agent(agent_instance.agent, test_cases)

print(f"准确率: {report['accuracy']:.1%}")
```

### 工具2: 导出验证失败的数据

```python
def export_failed_validations(input_file, output_file):
    """
    导出验证失败的数据，方便人工检查
    
    Args:
        input_file: SFT数据文件
        output_file: 输出文件
    """
    from sql_extractor_and_validator import SFTDataValidator
    import json
    
    validator = SFTDataValidator()
    
    failed_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            validation = validator.validate_sft_item(item)
            
            if not validation['is_valid']:
                failed_items.append({
                    'item': item,
                    'issues': validation['issues'],
                    'extracted_sqls': validation['extracted_sqls']
                })
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)
    
    print(f"导出了 {len(failed_items)} 条失败数据到 {output_file}")

# 使用
export_failed_validations("finetune_alpaca.jsonl", "failed_validations.json")
```

### 工具3: SQL执行日志分析

```python
def analyze_sql_execution_log():
    """
    分析sql.log，提取所有执行的SQL和结果
    """
    import json
    
    sqls = []
    with open('sql.log', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                if log.get('sql'):
                    sqls.append({
                        'timestamp': log['timestamp'],
                        'sql': log['sql'],
                        'status': log['status'],
                        'result': log.get('result')
                    })
            except:
                continue
    
    # 统计
    total = len(sqls)
    success = sum(1 for s in sqls if s['status'] == 'success')
    
    print(f"总SQL执行: {total}")
    print(f"成功: {success} ({success/total*100:.1f}%)")
    
    return sqls
```

---

## 🐛 常见问题

### Q1: 提取不到SQL

**原因**: Agent输出格式不标准

**解决**: 检查输出格式，或自定义提取规则

```python
# 打印Agent原始输出
result = agent.invoke({"input": question})
print(result['output'])

# 查看中间步骤
for step in result.get('intermediate_steps', []):
    print(step)
```

### Q2: 文本相似度低但结果正确

**原因**: SQL写法不同但语义等价

**解决**: 使用执行结果比较而非文本比较

```python
comparison = comparator.compare(
    sql1, sql2,
    check_text=False,      # 不比较文本
    check_results=True     # 只比较结果
)
```

### Q3: 标准答案执行失败

**原因**: 标准答案本身有问题

**解决**: 先验证标准答案

```python
comparator = SQLComparator()

# 验证标准答案能否执行
try:
    cursor = comparator.conn.cursor()
    cursor.execute(ground_truth_sql)
    result = cursor.fetchall()
    print(f"✅ 标准答案可执行，返回 {len(result)} 行")
except Exception as e:
    print(f"❌ 标准答案有误: {e}")
```

---

## 📁 文件说明

| 文件 | 功能 | 使用场景 |
|------|------|----------|
| `sql_extractor_and_validator.py` | 核心工具类 | 在代码中导入使用 |
| `example_sql_extraction_and_comparison.py` | 完整示例 | 学习如何使用 |
| `SQL提取和比较使用指南.md` | 本文档 | 查看使用方法 |

---

## 🎯 推荐工作流

### 开发阶段

```python
# 1. 小规模测试
test_questions = ["有多少个员工？", "有哪些部门？"]

for question in test_questions:
    result = compare_agent_sql_with_ground_truth(
        agent, question, ground_truths[question]
    )
    print(f"{question}: {'✅' if result['comparison']['is_correct'] else '❌'}")
```

### 数据生成阶段

```python
# 2. 实时验证
valid_data = []
for question in all_questions:
    result = agent.query(question)
    sqls = extract_sql(result)
    
    if sqls and validate_sql(sqls[-1]):
        valid_data.append(result)
```

### 质量检查阶段

```python
# 3. 批量验证
validator = SFTDataValidator()
report = validator.validate_sft_file("finetune_alpaca.jsonl", "ground_truths.jsonl")

if report['correct_with_gt'] / report['with_ground_truth'] >= 0.85:
    print("✅ 数据质量达标")
else:
    print("❌ 需要改进")
```

---

## 📚 相关资源

- [sql_agent/sql_agent_with_logging.py](sql_agent/sql_agent_with_logging.py) - SQL日志记录
- [callbacks.py](callbacks.py) - LLM回调处理
- [db_utils.py](db_utils.py) - 数据库工具

---

## ✅ 总结

使用这套工具，你可以：

1. ✅ **准确提取** - 从Agent输出中提取SQL
2. ✅ **精确比较** - 与标准答案进行文本和结果双重比较
3. ✅ **批量验证** - 自动验证大量SFT数据
4. ✅ **质量保证** - 确保只使用高质量数据进行微调

**核心优势**：
- 🎯 提取准确：支持多种格式
- 🔍 比较全面：文本+执行结果
- 🚀 使用简单：3行代码即可开始
- 📊 报告详细：清晰的验证报告

祝你的模型微调顺利！🎉


