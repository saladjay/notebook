# SQL验证工具 - 快速索引

## 🎯 核心问题

在制作SFT训练数据时：
1. 如何**提取Agent最后执行的SQL**？
2. 如何**与人工标准答案进行比较**？
3. 如何**验证SQL是否正确**？

---

## 🚀 快速开始（3行代码）

### 场景1: 提取Agent执行的SQL

```python
from sql_extractor_and_validator import extract_sql_from_agent
from sql_agent.sql_agent_base import SQLAgentBase

agent = SQLAgentBase()
sqls = extract_sql_from_agent(agent.agent, "有多少个员工？")
print(sqls)  # ['SELECT COUNT(*) FROM employees']
```

### 场景2: 与标准答案比较

```python
from sql_extractor_and_validator import compare_agent_sql_with_ground_truth
from sql_agent.sql_agent_base import SQLAgentBase

agent = SQLAgentBase()
result = compare_agent_sql_with_ground_truth(
    agent.agent,
    question="有多少个员工？",
    ground_truth_sql="SELECT COUNT(*) FROM employees"
)

print(f"是否正确: {result['comparison']['is_correct']}")  # True/False
print(f"文本相似度: {result['comparison']['text_similarity']:.2%}")  # 100%
```

### 场景3: 批量验证SFT文件

```python
from sql_extractor_and_validator import SFTDataValidator

validator = SFTDataValidator()
report = validator.validate_sft_file(
    "finetune_alpaca.jsonl",
    ground_truth_file="ground_truth_sqls.jsonl"  # 可选
)

print(f"有效数据: {report['valid_items']}/{report['total_items']}")
print(f"与标准答案一致: {report['correct_with_gt']}/{report['with_ground_truth']}")
```

---

## 📁 文件结构

```
SQLAGENT/
├── sql_extractor_and_validator.py       # 核心工具（必读）
├── example_sql_extraction_and_comparison.py  # 完整示例（推荐）
├── SQL提取和比较使用指南.md              # 详细文档（必读）
└── SQL验证工具README.md                  # 本文件
```

---

## 🛠️ 核心工具类

### 1. SQLExtractor - SQL提取器

**功能**: 从各种来源提取SQL语句

**方法**:
- `extract_from_text(text)` - 从文本中提取
- `extract_from_agent_output(output)` - 从Agent输出提取
- `extract_from_callback_history(callback)` - 从回调历史提取

**示例**:
```python
from sql_extractor_and_validator import SQLExtractor

extractor = SQLExtractor()
sqls = extractor.extract_from_text("""
    Action: sql_db_query
    Action Input: SELECT * FROM employees
""")
print(sqls)  # ['SELECT * FROM employees']
```

---

### 2. SQLComparator - SQL比较器

**功能**: 比较生成的SQL与标准答案

**方法**:
- `normalize_sql(sql)` - 标准化SQL
- `text_similarity(sql1, sql2)` - 计算文本相似度
- `execute_and_compare_results(sql1, sql2)` - 比较执行结果
- `compare(sql1, sql2)` - 完整比较

**示例**:
```python
from sql_extractor_and_validator import SQLComparator

comparator = SQLComparator()
result = comparator.compare(
    "SELECT COUNT(*) FROM employees",
    "SELECT COUNT(*) FROM employees"
)

print(result['is_correct'])  # True
print(result['text_similarity'])  # 1.0
print(result['results_comparison']['results_match'])  # True
```

---

### 3. SFTDataValidator - SFT数据验证器

**功能**: 批量验证SFT训练数据

**方法**:
- `validate_sft_item(item, ground_truth)` - 验证单个数据项
- `validate_sft_file(file_path, ground_truth_file)` - 验证整个文件

**示例**:
```python
from sql_extractor_and_validator import SFTDataValidator

validator = SFTDataValidator()

# 验证单个数据项
item = {
    "instruction": "根据数据库回答问题",
    "input": "有多少个员工？",
    "output": "Action Input: SELECT COUNT(*) FROM employees\nFinal Answer: 10名员工"
}
result = validator.validate_sft_item(item, "SELECT COUNT(*) FROM employees")
print(result['is_valid'])  # True

# 批量验证
report = validator.validate_sft_file("finetune_alpaca.jsonl")
```

---

## 📊 验证报告示例

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

================================================================================
❌ 无效数据详情（前10条）:
================================================================================

行 23:
  问题: 查询所有员工
  - SQL执行失败: no such table: employee
  文本相似度: 95.00%

行 45:
  问题: 统计部门数量
  - SQL执行结果不匹配
  文本相似度: 88.00%
```

---

## 🔄 完整工作流程

### 步骤1: 准备标准答案

```python
# 创建 ground_truth_sqls.jsonl
from example_sql_extraction_and_comparison import example4_create_ground_truth_file

example4_create_ground_truth_file()
```

生成的文件格式：
```json
{"question": "有多少个员工？", "sql": "SELECT COUNT(*) FROM employees"}
{"question": "有哪些部门？", "sql": "SELECT DISTINCT department FROM employees"}
```

### 步骤2: 生成SFT数据（实时验证）

```python
from sql_extractor_and_validator import SQLComparator, SQLExtractor
from sql_agent.sql_agent_base import SQLAgentBase

agent = SQLAgentBase()
comparator = SQLComparator()
extractor = SQLExtractor()

# 标准答案
ground_truths = {
    "有多少个员工？": "SELECT COUNT(*) FROM employees"
}

valid_data = []

for question in questions:
    # Agent生成回答
    result = agent.agent.invoke({"input": question})
    
    # 提取SQL
    sqls = extractor.extract_from_agent_output(result)
    
    if sqls and question in ground_truths:
        # 与标准答案比较
        comparison = comparator.compare(sqls[-1], ground_truths[question])
        
        if comparison['is_correct']:
            # 只保存正确的数据
            valid_data.append({
                "instruction": "根据数据库回答问题",
                "input": question,
                "output": result['output']
            })
            print(f"✅ {question}")
        else:
            print(f"❌ {question}: SQL不正确")

# 保存
import json
with open('finetune_alpaca_validated.jsonl', 'w', encoding='utf-8') as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

### 步骤3: 批量验证已生成的数据

```python
from sql_extractor_and_validator import SFTDataValidator

validator = SFTDataValidator()
report = validator.validate_sft_file(
    "finetune_alpaca.jsonl",
    ground_truth_file="ground_truth_sqls.jsonl"
)

# 检查质量
if report['correct_with_gt'] / report['with_ground_truth'] >= 0.85:
    print("✅ 数据质量良好，可以用于微调")
else:
    print("❌ 数据质量不佳，需要改进")
```

---

## 💡 高级用法

### 1. 批量测试Agent准确率

```python
from example_sql_extraction_and_comparison import example7_batch_test_agent

results = example7_batch_test_agent()
```

### 2. 只验证SQL能否执行（无标准答案）

```python
validator = SFTDataValidator()
result = validator.validate_sft_item(item, ground_truth=None)
# 只检查SQL能否执行，不比较结果
```

### 3. 语义等价性检查

```python
# 这两个SQL写法不同但结果相同
sql1 = "SELECT * FROM employees WHERE salary > 5000"
sql2 = "SELECT * FROM employees WHERE 5000 < salary"

comparator = SQLComparator()
result = comparator.compare(sql1, sql2, check_results=True)

print(result['is_correct'])  # True（结果相同）
print(result['text_similarity'])  # 可能不是100%
```

---

## 🎓 学习路径

### 新手入门

1. **阅读**: [SQL提取和比较使用指南.md](SQL提取和比较使用指南.md)
2. **运行**: `python example_sql_extraction_and_comparison.py`
3. **实践**: 修改示例代码，测试你的数据

### 进阶使用

1. **集成**: 将验证集成到数据生成流程
2. **定制**: 根据需求自定义提取和比较规则
3. **优化**: 根据验证结果优化Agent配置

---

## 📈 质量指标

| 指标 | 计算方法 | 良好阈值 |
|------|----------|----------|
| SQL提取率 | 包含SQL的数据 / 总数据 | ≥ 95% |
| SQL有效性 | 能执行的SQL / 包含SQL的数据 | ≥ 90% |
| 标准答案一致率 | 与ground truth一致 / 有ground truth的数据 | ≥ 85% |

---

## 🐛 常见问题

### Q1: 提取不到SQL？

**检查Agent输出格式**:
```python
result = agent.invoke({"input": "问题"})
print(result['output'])  # 查看原始输出
```

### Q2: 文本相似度低但结果正确？

**使用结果比较而非文本比较**:
```python
comparison = comparator.compare(sql1, sql2, check_text=False, check_results=True)
```

### Q3: 如何查看详细的比较结果？

```python
result = compare_agent_sql_with_ground_truth(agent, question, gt_sql)

print("提取的所有SQL:", result['extracted_sqls'])
print("最终SQL:", result['final_sql'])
print("比较详情:", result['comparison'])
if result['comparison']['results_comparison']:
    print("执行结果:", result['comparison']['results_comparison'])
```

---

## 📞 获取帮助

1. 查看 [SQL提取和比较使用指南.md](SQL提取和比较使用指南.md) - 详细文档
2. 运行 `python example_sql_extraction_and_comparison.py` - 交互式示例
3. 查看代码注释和docstring

---

## ✅ 核心优势

- ✅ **提取准确** - 支持多种格式的SQL提取
- ✅ **比较全面** - 文本相似度 + 执行结果双重验证
- ✅ **使用简单** - 3行代码即可开始
- ✅ **批量处理** - 支持大规模数据验证
- ✅ **报告详细** - 清晰的验证报告和错误详情

---

**开始使用**: `python example_sql_extraction_and_comparison.py`

**快速测试**:
```python
from sql_extractor_and_validator import extract_sql_from_agent
from sql_agent.sql_agent_base import SQLAgentBase

agent = SQLAgentBase()
sqls = extract_sql_from_agent(agent.agent, "有多少个员工？")
print(sqls)
```

祝你的模型微调顺利！🚀


