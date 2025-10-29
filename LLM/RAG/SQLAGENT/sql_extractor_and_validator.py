"""
SQL提取和验证工具
用于：
1. 从Agent输出中提取执行的SQL
2. 与人工标准答案进行比较
3. 验证SQL的正确性
"""
import re
import json
import sqlite3
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher
from config import get_config
from db_utils import connect_sqlite_safely, get_db_path_from_uri


class SQLExtractor:
    """从Agent输出中提取SQL语句"""
    
    @staticmethod
    def extract_from_text(text: str) -> List[str]:
        """
        从文本中提取所有SQL语句
        
        Args:
            text: 包含SQL的文本
            
        Returns:
            SQL语句列表
        """
        sql_list = []
        
        # 方法1: 提取代码块中的SQL
        code_blocks = re.findall(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        sql_list.extend(code_blocks)
        
        # 方法2: 提取SELECT语句
        select_statements = re.findall(
            r'(SELECT\s+.*?(?:;|\n\n|$))', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        sql_list.extend(select_statements)
        
        # 方法3: 从Action Input中提取
        action_inputs = re.findall(
            r'Action\s+Input:\s*["\']?(.*?)["\']?\s*(?:Observation:|$)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        for inp in action_inputs:
            if 'SELECT' in inp.upper():
                sql_list.append(inp.strip())
        
        # 去重并清理
        unique_sql = []
        seen = set()
        for sql in sql_list:
            sql_clean = sql.strip().rstrip(';').strip()
            sql_upper = sql_clean.upper()
            if sql_clean and sql_upper not in seen and 'SELECT' in sql_upper:
                seen.add(sql_upper)
                unique_sql.append(sql_clean)
        
        return unique_sql
    
    @staticmethod
    def extract_from_agent_output(agent_output: Dict) -> List[str]:
        """
        从Agent的输出字典中提取SQL
        
        Args:
            agent_output: Agent.invoke()的返回值
            
        Returns:
            SQL语句列表
        """
        sql_list = []
        
        # 从output字段提取
        if 'output' in agent_output:
            sql_list.extend(SQLExtractor.extract_from_text(agent_output['output']))
        
        # 从intermediate_steps提取
        if 'intermediate_steps' in agent_output:
            for step in agent_output['intermediate_steps']:
                if isinstance(step, tuple) and len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(action, 'tool_input'):
                        sql_list.extend(SQLExtractor.extract_from_text(str(action.tool_input)))
        
        return sql_list
    
    @staticmethod
    def extract_from_callback_history(callback) -> List[Dict]:
        """
        从LLMLoggingCallback的历史中提取SQL
        
        Args:
            callback: LLMLoggingCallback实例
            
        Returns:
            包含SQL信息的字典列表
        """
        sql_records = []
        
        if hasattr(callback, 'interactions'):
            for interaction in callback.interactions:
                # 从generations中提取
                if 'generations' in interaction:
                    for gen in interaction['generations']:
                        sqls = SQLExtractor.extract_from_text(gen)
                        for sql in sqls:
                            sql_records.append({
                                'timestamp': interaction.get('timestamp'),
                                'sql': sql,
                                'source': 'llm_generation',
                                'full_text': gen
                            })
        
        return sql_records


class SQLComparator:
    """SQL比较器 - 比较生成的SQL与标准答案"""
    
    def __init__(self, db_uri: str = None):
        """
        初始化比较器
        
        Args:
            db_uri: 数据库URI，用于执行SQL
        """
        if db_uri is None:
            config = get_config()
            db_uri = config['database_uri']
        
        self.db_uri = db_uri
        self.conn = connect_sqlite_safely(db_uri)
    
    def __del__(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def normalize_sql(self, sql: str) -> str:
        """
        标准化SQL语句，方便比较
        
        Args:
            sql: SQL语句
            
        Returns:
            标准化后的SQL
        """
        # 移除多余空白
        sql = re.sub(r'\s+', ' ', sql)
        # 转大写
        sql = sql.upper()
        # 移除末尾分号
        sql = sql.rstrip(';').strip()
        return sql
    
    def text_similarity(self, sql1: str, sql2: str) -> float:
        """
        计算两个SQL的文本相似度（0-1）
        
        Args:
            sql1: SQL语句1
            sql2: SQL语句2
            
        Returns:
            相似度分数
        """
        norm1 = self.normalize_sql(sql1)
        norm2 = self.normalize_sql(sql2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def execute_and_compare_results(
        self, 
        sql1: str, 
        sql2: str
    ) -> Dict:
        """
        执行两个SQL并比较结果
        
        Args:
            sql1: 生成的SQL
            sql2: 标准答案SQL
            
        Returns:
            比较结果字典
        """
        result = {
            'sql1': sql1,
            'sql2': sql2,
            'sql1_success': False,
            'sql2_success': False,
            'results_match': False,
            'sql1_result': None,
            'sql2_result': None,
            'error1': None,
            'error2': None
        }
        
        # 执行SQL1
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql1)
            result['sql1_result'] = cursor.fetchall()
            result['sql1_success'] = True
        except Exception as e:
            result['error1'] = str(e)
        
        # 执行SQL2
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql2)
            result['sql2_result'] = cursor.fetchall()
            result['sql2_success'] = True
        except Exception as e:
            result['error2'] = str(e)
        
        # 比较结果
        if result['sql1_success'] and result['sql2_success']:
            # 转换为集合进行比较（忽略顺序）
            set1 = {tuple(row) for row in result['sql1_result']}
            set2 = {tuple(row) for row in result['sql2_result']}
            result['results_match'] = set1 == set2
        
        return result
    
    def compare(
        self, 
        generated_sql: str, 
        ground_truth_sql: str,
        check_text: bool = True,
        check_results: bool = True
    ) -> Dict:
        """
        完整的SQL比较
        
        Args:
            generated_sql: 生成的SQL
            ground_truth_sql: 标准答案SQL
            check_text: 是否检查文本相似度
            check_results: 是否检查执行结果
            
        Returns:
            完整的比较结果
        """
        comparison = {
            'generated_sql': generated_sql,
            'ground_truth_sql': ground_truth_sql,
            'is_correct': False,
            'text_similarity': None,
            'results_comparison': None,
            'issues': []
        }
        
        # 1. 文本相似度比较
        if check_text:
            comparison['text_similarity'] = self.text_similarity(
                generated_sql, 
                ground_truth_sql
            )
            
            # 如果文本完全相同或非常相似，认为正确
            if comparison['text_similarity'] >= 0.95:
                comparison['is_correct'] = True
                return comparison
        
        # 2. 执行结果比较
        if check_results:
            comparison['results_comparison'] = self.execute_and_compare_results(
                generated_sql,
                ground_truth_sql
            )
            
            # 检查是否有错误
            if not comparison['results_comparison']['sql1_success']:
                comparison['issues'].append(f"生成的SQL执行失败: {comparison['results_comparison']['error1']}")
            
            if not comparison['results_comparison']['sql2_success']:
                comparison['issues'].append(f"标准答案SQL执行失败: {comparison['results_comparison']['error2']}")
            
            # 如果都执行成功，比较结果
            if (comparison['results_comparison']['sql1_success'] and 
                comparison['results_comparison']['sql2_success']):
                
                if comparison['results_comparison']['results_match']:
                    comparison['is_correct'] = True
                else:
                    comparison['issues'].append("SQL执行结果不匹配")
        
        return comparison


class SFTDataValidator:
    """SFT数据验证器 - 验证训练数据中的SQL"""
    
    def __init__(self, db_uri: str = None):
        """
        初始化验证器
        
        Args:
            db_uri: 数据库URI
        """
        self.extractor = SQLExtractor()
        self.comparator = SQLComparator(db_uri)
    
    def validate_sft_item(
        self, 
        item: Dict, 
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        验证单个SFT数据项
        
        Args:
            item: SFT数据项（包含instruction, input, output字段）
            ground_truth: 标准答案SQL（可选）
            
        Returns:
            验证结果
        """
        result = {
            'item': item,
            'extracted_sqls': [],
            'has_sql': False,
            'is_valid': False,
            'comparison_result': None,
            'issues': []
        }
        
        # 从output中提取SQL
        output = item.get('output', '')
        sqls = self.extractor.extract_from_text(output)
        result['extracted_sqls'] = sqls
        
        if not sqls:
            result['issues'].append('未找到SQL语句')
            return result
        
        result['has_sql'] = True
        
        # 取最后一个SQL（通常是实际执行的）
        final_sql = sqls[-1]
        
        # 如果提供了ground truth，进行比较
        if ground_truth:
            comparison = self.comparator.compare(
                final_sql,
                ground_truth,
                check_text=True,
                check_results=True
            )
            result['comparison_result'] = comparison
            result['is_valid'] = comparison['is_correct']
            result['issues'].extend(comparison['issues'])
        else:
            # 如果没有ground truth，只验证SQL能否执行
            try:
                cursor = self.comparator.conn.cursor()
                cursor.execute(final_sql)
                cursor.fetchall()
                result['is_valid'] = True
            except Exception as e:
                result['is_valid'] = False
                result['issues'].append(f'SQL执行失败: {str(e)}')
        
        return result
    
    def validate_sft_file(
        self, 
        file_path: str,
        ground_truth_file: Optional[str] = None
    ) -> Dict:
        """
        验证整个SFT数据文件
        
        Args:
            file_path: SFT数据文件路径
            ground_truth_file: 标准答案文件路径（可选）
                             格式: {"question": "...", "sql": "..."}
            
        Returns:
            验证报告
        """
        print(f"\n{'='*80}")
        print(f"🔍 验证SFT数据文件")
        print(f"{'='*80}")
        print(f"文件: {file_path}")
        
        # 加载ground truth
        ground_truths = {}
        if ground_truth_file and Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        gt = json.loads(line.strip())
                        question = gt.get('question', '')
                        sql = gt.get('sql', '')
                        if question and sql:
                            ground_truths[question] = sql
                    except:
                        continue
            print(f"标准答案: {ground_truth_file} ({len(ground_truths)}条)")
        print(f"{'='*80}\n")
        
        report = {
            'file': file_path,
            'total_items': 0,
            'has_sql': 0,
            'valid_items': 0,
            'invalid_items': 0,
            'with_ground_truth': 0,
            'correct_with_gt': 0,
            'validation_details': []
        }
        
        # 验证每一行
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    report['total_items'] += 1
                    
                    # 获取ground truth
                    question = item.get('input', '')
                    gt_sql = ground_truths.get(question)
                    if gt_sql:
                        report['with_ground_truth'] += 1
                    
                    # 验证
                    validation = self.validate_sft_item(item, gt_sql)
                    
                    if validation['has_sql']:
                        report['has_sql'] += 1
                    
                    if validation['is_valid']:
                        report['valid_items'] += 1
                        if gt_sql:
                            report['correct_with_gt'] += 1
                    else:
                        report['invalid_items'] += 1
                        report['validation_details'].append({
                            'line': i,
                            'question': question,
                            'validation': validation
                        })
                    
                    # 打印进度
                    if i % 100 == 0:
                        print(f"已验证 {i} 条数据...")
                
                except json.JSONDecodeError:
                    print(f"⚠️  行 {i}: JSON解析错误")
                    report['invalid_items'] += 1
        
        # 打印报告
        self._print_report(report)
        
        return report
    
    def _print_report(self, report: Dict):
        """打印验证报告"""
        print(f"\n{'='*80}")
        print("📊 验证报告")
        print(f"{'='*80}")
        print(f"文件: {report['file']}")
        print(f"总数据量: {report['total_items']}")
        print(f"\n📈 SQL提取:")
        print(f"  包含SQL: {report['has_sql']} ({report['has_sql']/report['total_items']*100:.1f}%)" if report['total_items'] > 0 else "")
        
        print(f"\n✅ SQL有效性:")
        print(f"  有效SQL: {report['valid_items']} ({report['valid_items']/report['has_sql']*100:.1f}%)" if report['has_sql'] > 0 else "")
        print(f"  无效SQL: {report['invalid_items']}")
        
        if report['with_ground_truth'] > 0:
            print(f"\n🎯 与标准答案比较:")
            print(f"  有标准答案: {report['with_ground_truth']}")
            print(f"  与标准答案一致: {report['correct_with_gt']} ({report['correct_with_gt']/report['with_ground_truth']*100:.1f}%)")
        
        if report['validation_details']:
            print(f"\n{'='*80}")
            print(f"❌ 无效数据详情（前10条）:")
            print(f"{'='*80}")
            for detail in report['validation_details'][:10]:
                print(f"\n行 {detail['line']}:")
                print(f"  问题: {detail['question']}")
                for issue in detail['validation']['issues']:
                    print(f"  - {issue}")
                if detail['validation'].get('comparison_result'):
                    comp = detail['validation']['comparison_result']
                    if comp.get('text_similarity'):
                        print(f"  文本相似度: {comp['text_similarity']:.2%}")
        
        print(f"\n{'='*80}\n")


# ============================================
# 使用示例和工具函数
# ============================================

def extract_sql_from_agent(agent, question: str) -> List[str]:
    """
    执行Agent查询并提取SQL
    
    Args:
        agent: SQL Agent实例
        question: 用户问题
        
    Returns:
        提取的SQL列表
    """
    from callbacks import LLMLoggingCallback
    
    # 创建回调
    callback = LLMLoggingCallback(
        log_to_console=False,
        log_to_file=False,
        save_json=False
    )
    
    # 执行查询
    result = agent.invoke(
        {"input": question},
        config={"callbacks": [callback]}
    )
    
    # 提取SQL
    extractor = SQLExtractor()
    sqls = extractor.extract_from_agent_output(result)
    sqls.extend(extractor.extract_from_callback_history(callback))
    
    # 去重
    unique_sqls = []
    seen = set()
    for sql in sqls:
        sql_norm = re.sub(r'\s+', ' ', sql.upper())
        if sql_norm not in seen:
            seen.add(sql_norm)
            unique_sqls.append(sql)
    
    return unique_sqls


def compare_agent_sql_with_ground_truth(
    agent,
    question: str,
    ground_truth_sql: str
) -> Dict:
    """
    执行Agent查询，提取SQL并与标准答案比较
    
    Args:
        agent: SQL Agent实例
        question: 用户问题
        ground_truth_sql: 标准答案SQL
        
    Returns:
        比较结果
    """
    # 提取SQL
    sqls = extract_sql_from_agent(agent, question)
    
    if not sqls:
        return {
            'question': question,
            'ground_truth': ground_truth_sql,
            'extracted_sqls': [],
            'comparison': None,
            'error': '未能提取SQL'
        }
    
    # 取最后一个SQL（通常是实际执行的）
    final_sql = sqls[-1]
    
    # 比较
    comparator = SQLComparator()
    comparison = comparator.compare(final_sql, ground_truth_sql)
    
    return {
        'question': question,
        'ground_truth': ground_truth_sql,
        'extracted_sqls': sqls,
        'final_sql': final_sql,
        'comparison': comparison
    }


if __name__ == '__main__':
    """测试和演示"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🔍 SQL提取和验证工具                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 测试1: SQL提取
    print("\n" + "="*80)
    print("测试1: 从文本中提取SQL")
    print("="*80)
    
    test_text = """
    Thought: 我需要查询所有员工
    Action: sql_db_query
    Action Input: SELECT * FROM employees WHERE salary > 5000
    Observation: 返回了10条记录
    
    ```sql
    SELECT COUNT(*) FROM employees
    ```
    
    Final Answer: 共有10名员工薪资超过5000
    """
    
    extractor = SQLExtractor()
    sqls = extractor.extract_from_text(test_text)
    print(f"提取到 {len(sqls)} 条SQL:")
    for i, sql in enumerate(sqls, 1):
        print(f"{i}. {sql}")
    
    # 测试2: SQL比较
    print("\n" + "="*80)
    print("测试2: SQL比较")
    print("="*80)
    
    config = get_config()
    comparator = SQLComparator(config['database_uri'])
    
    sql1 = "SELECT * FROM employees"
    sql2 = "SELECT * FROM employees"
    
    result = comparator.compare(sql1, sql2)
    print(f"SQL1: {sql1}")
    print(f"SQL2: {sql2}")
    print(f"文本相似度: {result['text_similarity']:.2%}")
    print(f"是否正确: {result['is_correct']}")
    
    print("\n" + "="*80)
    print("✅ 测试完成")
    print("="*80)

