"""
单元测试
测试SQL Agent的各个组件
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from intent_router import IntentRouter, IntentClassification
from sql_generator import SQLGenerator, SQLQueryResult
from query_executor import QueryExecutor
from security import SecurityModule, SecurityLevel
from memory import MemorySystem
from planner import Planner, ExecutionPlan, SubTask


class TestIntentRouter(unittest.TestCase):
    """测试意图路由器"""
    
    def setUp(self):
        self.mock_llm = Mock()
        self.router = IntentRouter(self.mock_llm)
    
    def test_classification_types(self):
        """测试分类类型判断"""
        # 简单查询
        simple = IntentClassification(
            intent="simple_query",
            reasoning="这是一个简单查询",
            confidence=0.9
        )
        self.assertTrue(self.router.is_simple_query(simple))
        self.assertFalse(self.router.is_complex_query(simple))
        self.assertFalse(self.router.should_clarify(simple))
        
        # 复杂查询
        complex_query = IntentClassification(
            intent="complex_query",
            reasoning="需要多步骤",
            confidence=0.85
        )
        self.assertTrue(self.router.is_complex_query(complex_query))
        self.assertFalse(self.router.is_simple_query(complex_query))
        
        # 需要澄清
        clarification = IntentClassification(
            intent="clarification_needed",
            reasoning="问题模糊",
            confidence=0.95
        )
        self.assertTrue(self.router.should_clarify(clarification))


class TestSecurityModule(unittest.TestCase):
    """测试安全模块"""
    
    def test_validate_safe_sql(self):
        """测试安全的SQL"""
        security = SecurityModule(SecurityLevel.MEDIUM)
        
        safe_sql = "SELECT * FROM users WHERE id = 1"
        is_safe, error = security.validate_sql(safe_sql)
        self.assertTrue(is_safe)
        self.assertIsNone(error)
    
    def test_validate_dangerous_sql(self):
        """测试危险的SQL"""
        security = SecurityModule(SecurityLevel.MEDIUM)
        
        # DROP语句
        dangerous_sql = "DROP TABLE users"
        is_safe, error = security.validate_sql(dangerous_sql)
        self.assertFalse(is_safe)
        self.assertIn("DROP", error)
        
        # SQL注入
        injection_sql = "SELECT * FROM users WHERE id = 1 OR 1=1"
        is_safe, error = security.validate_sql(injection_sql)
        self.assertFalse(is_safe)
    
    def test_add_limit_clause(self):
        """测试添加LIMIT子句"""
        security = SecurityModule(SecurityLevel.MEDIUM)
        security.set_max_rows(100)
        
        # 没有LIMIT的SQL
        sql = "SELECT * FROM users"
        limited_sql = security.add_limit_clause(sql)
        self.assertIn("LIMIT 100", limited_sql)
        
        # 已有LIMIT的SQL
        sql_with_limit = "SELECT * FROM users LIMIT 50"
        result = security.add_limit_clause(sql_with_limit)
        self.assertIn("LIMIT", result)
    
    def test_security_levels(self):
        """测试不同安全级别"""
        # 高安全级别
        high_security = SecurityModule(SecurityLevel.HIGH)
        update_sql = "UPDATE users SET name = 'test'"
        is_safe, _ = high_security.validate_sql(update_sql)
        self.assertFalse(is_safe)
        
        # 低安全级别
        low_security = SecurityModule(SecurityLevel.LOW)
        is_safe, _ = low_security.validate_sql(update_sql)
        self.assertTrue(is_safe)  # 低安全级别允许UPDATE


class TestMemorySystem(unittest.TestCase):
    """测试记忆系统"""
    
    def test_add_conversation(self):
        """测试添加对话"""
        memory = MemorySystem()
        
        memory.add_conversation("Hello", "Hi there!")
        memory.add_conversation("How are you?", "I'm fine")
        
        recent = memory.get_recent_conversations(n=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["user"], "Hello")
    
    def test_add_sql_query(self):
        """测试添加SQL查询"""
        memory = MemorySystem()
        
        memory.add_sql_query(
            question="Get all users",
            sql="SELECT * FROM users",
            result="100 rows",
            success=True
        )
        
        queries = memory.get_recent_sql_queries(n=1)
        self.assertEqual(len(queries), 1)
        self.assertTrue(queries[0]["success"])
    
    def test_get_successful_queries(self):
        """测试获取成功的查询"""
        memory = MemorySystem()
        
        # 添加一些查询
        memory.add_sql_query("Q1", "SQL1", "R1", True)
        memory.add_sql_query("Q2", "SQL2", "R2", False)
        memory.add_sql_query("Q3", "SQL3", "R3", True)
        
        successful = memory.get_successful_sql_queries(n=10)
        self.assertEqual(len(successful), 2)
        self.assertTrue(all(q["success"] for q in successful))
    
    def test_knowledge_storage(self):
        """测试知识存储"""
        memory = MemorySystem()
        
        memory.save_knowledge("user_preference", "natural_language")
        memory.save_knowledge("output_format", "table")
        
        self.assertEqual(memory.get_knowledge("user_preference"), "natural_language")
        self.assertEqual(memory.get_knowledge("nonexistent", "default"), "default")
    
    def test_memory_summary(self):
        """测试记忆摘要"""
        memory = MemorySystem()
        
        memory.add_conversation("Q1", "A1")
        memory.add_sql_query("Q1", "SQL1", "R1", True)
        memory.save_knowledge("key", "value")
        
        summary = memory.get_summary()
        self.assertEqual(summary["total_conversations"], 1)
        self.assertEqual(summary["total_sql_queries"], 1)
        self.assertEqual(summary["knowledge_items"], 1)
    
    def test_clear_memory(self):
        """测试清空记忆"""
        memory = MemorySystem()
        
        memory.add_conversation("Q", "A")
        memory.add_sql_query("Q", "SQL", "R", True)
        memory.save_knowledge("k", "v")
        
        memory.clear()
        
        summary = memory.get_summary()
        self.assertEqual(summary["total_conversations"], 0)
        self.assertEqual(summary["total_sql_queries"], 0)
        self.assertEqual(summary["knowledge_items"], 0)


class TestQueryExecutor(unittest.TestCase):
    """测试查询执行器"""
    
    def test_format_empty_results(self):
        """测试格式化空结果"""
        executor = QueryExecutor("sqlite:///:memory:")
        
        empty_df = pd.DataFrame()
        result = executor.format_results(empty_df)
        self.assertIn("未返回任何结果", result)
    
    def test_format_results_as_table(self):
        """测试表格格式"""
        executor = QueryExecutor("sqlite:///:memory:")
        
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        
        result = executor.format_results(df, format_type="table")
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
    
    def test_result_summary(self):
        """测试结果摘要"""
        executor = QueryExecutor("sqlite:///:memory:")
        
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [85, 90, 95]
        })
        
        summary = executor.get_result_summary(df)
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["column_count"], 2)
        self.assertIn("numeric_stats", summary)
        self.assertIn("score", summary["numeric_stats"])


class TestPlanner(unittest.TestCase):
    """测试规划器"""
    
    def test_get_next_task(self):
        """测试获取下一个任务"""
        planner = Planner(Mock())
        
        plan = ExecutionPlan(
            tasks=[
                SubTask(task_id=1, description="Task 1", depends_on=[], query_type="sql_query"),
                SubTask(task_id=2, description="Task 2", depends_on=[1], query_type="sql_query"),
                SubTask(task_id=3, description="Task 3", depends_on=[1], query_type="calculation"),
            ],
            reasoning="Test plan"
        )
        
        # 初始状态，应该返回任务1
        next_task = planner.get_next_task(plan, [])
        self.assertEqual(next_task.task_id, 1)
        
        # 完成任务1后，可以执行任务2或3
        next_task = planner.get_next_task(plan, [1])
        self.assertIn(next_task.task_id, [2, 3])
        
        # 所有任务完成
        next_task = planner.get_next_task(plan, [1, 2, 3])
        self.assertIsNone(next_task)
    
    def test_is_plan_complete(self):
        """测试计划是否完成"""
        planner = Planner(Mock())
        
        plan = ExecutionPlan(
            tasks=[
                SubTask(task_id=1, description="Task 1", depends_on=[], query_type="sql_query"),
                SubTask(task_id=2, description="Task 2", depends_on=[1], query_type="sql_query"),
            ],
            reasoning="Test"
        )
        
        self.assertFalse(planner.is_plan_complete(plan, []))
        self.assertFalse(planner.is_plan_complete(plan, [1]))
        self.assertTrue(planner.is_plan_complete(plan, [1, 2]))
    
    def test_task_results_context(self):
        """测试任务结果上下文"""
        planner = Planner(Mock())
        
        results = {
            1: "Result 1",
            2: "Result 2"
        }
        
        context = planner.get_task_results_context(results)
        self.assertIn("任务1", context)
        self.assertIn("Result 1", context)
        self.assertIn("任务2", context)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestIntentRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityModule))
    suite.addTests(loader.loadTestsFromTestCase(TestMemorySystem))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestPlanner))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)





