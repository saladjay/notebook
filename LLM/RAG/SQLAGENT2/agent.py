"""
SQL Agent核心模块
整合所有组件，实现完整的Agent逻辑
"""
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
import pandas as pd

from intent_router import IntentRouter, IntentClassification
from sql_generator import SQLGenerator
from query_executor import QueryExecutor
from planner import Planner, ExecutionPlan
from clarifier import Clarifier
from knowledge_base import KnowledgeBase
from security import SecurityModule, SecurityLevel
from memory import MemorySystem


class SQLAgent:
    """SQL Agent"""
    
    def __init__(
        self,
        database_uri: str,
        openai_api_key: str,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        verbose: bool = True
    ):
        """
        初始化SQL Agent
        
        Args:
            database_uri: 数据库连接URI
            openai_api_key: OpenAI API密钥
            model_name: 模型名称
            temperature: 温度参数
            security_level: 安全级别
            verbose: 是否显示详细信息
        """
        self.verbose = verbose
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key
        )
        
        # 初始化各个模块
        self.intent_router = IntentRouter(self.llm)
        self.sql_generator = SQLGenerator(self.llm)
        self.query_executor = QueryExecutor(database_uri)
        self.planner = Planner(self.llm)
        self.clarifier = Clarifier(self.llm)
        self.knowledge_base = KnowledgeBase(database_uri)
        self.security = SecurityModule(security_level)
        self.memory = MemorySystem()
        
        # 澄清状态
        self.pending_clarification: Optional[Dict[str, str]] = None
    
    def query(self, question: str) -> str:
        """
        处理用户查询
        
        Args:
            question: 用户的自然语言问题
            
        Returns:
            查询结果的自然语言描述
        """
        try:
            # 如果有待澄清的问题，处理澄清回答
            if self.pending_clarification:
                return self._handle_clarification_response(question)
            
            self._log(f"\n{'='*60}")
            self._log(f"用户问题: {question}")
            self._log(f"{'='*60}\n")
            
            # 1. 意图识别与路由
            classification = self._classify_intent(question)
            self._log(f"意图分类: {classification.intent}")
            self._log(f"置信度: {classification.confidence}")
            self._log(f"理由: {classification.reasoning}\n")
            
            # 2. 根据意图类型分发处理
            if self.intent_router.should_clarify(classification):
                return self._handle_clarification(question, classification)
            elif self.intent_router.is_simple_query(classification):
                return self._handle_simple_query(question)
            else:  # complex_query
                return self._handle_complex_query(question)
                
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            self._log(f"错误: {error_msg}")
            return error_msg
    
    def _classify_intent(self, question: str) -> IntentClassification:
        """分类用户意图"""
        tables = self.knowledge_base.get_table_names()
        return self.intent_router.classify_intent(question, tables)
    
    def _handle_simple_query(self, question: str) -> str:
        """处理简单查询"""
        self._log("处理简单查询...\n")
        
        # 获取上下文
        schema = self.knowledge_base.get_database_schema()
        context = self.knowledge_base.get_context_for_question(question)
        
        # 获取成功的历史查询作为示例
        examples = self.memory.get_successful_sql_queries(n=3)
        
        # 生成SQL
        self._log("生成SQL...")
        sql_result = self.sql_generator.generate_sql(
            question=question,
            schema=schema,
            context=context,
            examples=examples
        )
        
        self._log(f"生成的SQL:\n{sql_result.sql}\n")
        self._log(f"解释: {sql_result.explanation}\n")
        
        # 安全检查
        is_safe, error_msg = self.security.validate_sql(sql_result.sql)
        if not is_safe:
            self._log(f"安全检查失败: {error_msg}")
            return f"出于安全考虑，无法执行此查询: {error_msg}"
        
        # 添加LIMIT
        safe_sql = self.security.add_limit_clause(sql_result.sql)
        if safe_sql != sql_result.sql:
            self._log(f"已添加安全限制: LIMIT {self.security.max_rows}")
        
        # 执行查询
        self._log("执行查询...")
        success, result, error = self.query_executor.execute_query(safe_sql)
        
        if not success:
            self._log(f"查询执行失败: {error}\n")
            # 尝试修正SQL
            self._log("尝试修正SQL...")
            refined_sql = self.sql_generator.refine_sql(
                original_sql=safe_sql,
                error_message=error,
                schema=schema
            )
            
            # 重新执行
            success, result, error = self.query_executor.execute_query(refined_sql.sql)
            if not success:
                self.memory.add_sql_query(question, safe_sql, None, False)
                return f"查询执行失败: {error}"
        
        # 格式化结果
        formatted_result = self.query_executor.format_results(result)
        
        # 记录到记忆
        self.memory.add_sql_query(question, safe_sql, formatted_result, True)
        self.memory.add_conversation(question, formatted_result)
        
        self._log("查询完成!\n")
        return formatted_result
    
    def _handle_complex_query(self, question: str) -> str:
        """处理复杂多步查询"""
        self._log("处理复杂查询...\n")
        
        schema = self.knowledge_base.get_database_schema()
        
        # 创建执行计划
        self._log("创建执行计划...")
        plan = self.planner.create_plan(question, schema)
        
        self._log(f"规划理由: {plan.reasoning}")
        self._log(f"子任务数量: {len(plan.tasks)}\n")
        
        for task in plan.tasks:
            self._log(f"任务{task.task_id}: {task.description}")
            if task.depends_on:
                self._log(f"  依赖: {task.depends_on}")
        self._log("")
        
        # 执行计划
        completed_tasks: List[int] = []
        task_results: Dict[int, Any] = {}
        
        max_iterations = len(plan.tasks) * 2  # 防止无限循环
        iteration = 0
        
        while not self.planner.is_plan_complete(plan, completed_tasks):
            iteration += 1
            if iteration > max_iterations:
                return "执行计划超时，请简化问题后重试。"
            
            # 获取下一个任务
            next_task = self.planner.get_next_task(plan, completed_tasks)
            if not next_task:
                break
            
            self._log(f"\n执行任务{next_task.task_id}: {next_task.description}")
            
            # 构建任务上下文
            task_context = self.planner.get_task_results_context(task_results)
            
            # 执行任务
            if next_task.query_type == "sql_query":
                task_result = self._execute_sql_task(
                    next_task.description,
                    schema,
                    task_context
                )
            else:
                # 其他类型的任务可以扩展
                task_result = f"任务类型 {next_task.query_type} 暂不支持"
            
            task_results[next_task.task_id] = task_result
            completed_tasks.append(next_task.task_id)
            self._log(f"任务{next_task.task_id}完成")
        
        # 汇总结果
        final_result = self._summarize_complex_results(question, plan, task_results)
        
        # 记录到记忆
        self.memory.add_execution(question, final_result, {"plan": plan.dict()})
        self.memory.add_conversation(question, final_result)
        
        self._log("\n复杂查询完成!\n")
        return final_result
    
    def _execute_sql_task(self, task_description: str, schema: str, context: str) -> str:
        """执行SQL任务"""
        # 生成SQL
        sql_result = self.sql_generator.generate_sql(
            question=task_description,
            schema=schema,
            context=context
        )
        
        # 安全检查
        is_safe, error_msg = self.security.validate_sql(sql_result.sql)
        if not is_safe:
            return f"安全检查失败: {error_msg}"
        
        # 执行
        safe_sql = self.security.add_limit_clause(sql_result.sql)
        success, result, error = self.query_executor.execute_query(safe_sql)
        
        if not success:
            return f"执行失败: {error}"
        
        return self.query_executor.format_results(result)
    
    def _summarize_complex_results(
        self,
        question: str,
        plan: ExecutionPlan,
        task_results: Dict[int, Any]
    ) -> str:
        """汇总复杂查询的结果"""
        from langchain.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个数据分析助手。根据多个子任务的结果，生成对用户问题的完整回答。回答要清晰、准确、易懂。"),
            ("user", """用户问题: {question}

执行计划: {plan}

各任务结果:
{results}

请生成最终答案:""")
        ])
        
        results_text = ""
        for task_id, result in task_results.items():
            task = next(t for t in plan.tasks if t.task_id == task_id)
            results_text += f"\n任务{task_id} ({task.description}):\n{result}\n"
        
        chain = prompt | self.llm
        response = chain.invoke({
            "question": question,
            "plan": plan.reasoning,
            "results": results_text
        })
        
        return response.content
    
    def _handle_clarification(
        self,
        question: str,
        classification: IntentClassification
    ) -> str:
        """处理需要澄清的问题"""
        self._log("需要澄清用户需求...\n")
        
        schema = self.knowledge_base.get_database_schema()
        clarification_question = self.clarifier.generate_clarification(
            question=question,
            schema=schema,
            reasoning=classification.reasoning
        )
        
        # 保存待澄清状态
        self.pending_clarification = {
            "original_question": question,
            "clarification": clarification_question
        }
        
        return clarification_question
    
    def _handle_clarification_response(self, user_response: str) -> str:
        """处理用户的澄清回答"""
        if not self.pending_clarification:
            return "没有待澄清的问题。"
        
        # 生成完整问题
        complete_question = self.clarifier.process_clarification_response(
            original_question=self.pending_clarification["original_question"],
            clarification=self.pending_clarification["clarification"],
            user_response=user_response
        )
        
        # 清除待澄清状态
        self.pending_clarification = None
        
        self._log(f"完整问题: {complete_question}\n")
        
        # 重新处理完整问题
        return self.query(complete_question)
    
    def _log(self, message: str):
        """输出日志"""
        if self.verbose:
            print(message)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return self.memory.get_summary()
    
    def clear_memory(self):
        """清空记忆"""
        self.memory.clear()


