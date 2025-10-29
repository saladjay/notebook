"""
规划器模块
将复杂问题分解为多个子任务并规划执行顺序
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """子任务"""
    task_id: int = Field(description="任务ID")
    description: str = Field(description="任务描述")
    depends_on: List[int] = Field(default=[], description="依赖的任务ID列表")
    query_type: str = Field(description="查询类型：sql_query/calculation/aggregation")


class ExecutionPlan(BaseModel):
    """执行计划"""
    tasks: List[SubTask] = Field(description="子任务列表")
    reasoning: str = Field(description="规划理由")


class Planner:
    """规划器"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化规划器
        
        Args:
            llm: 语言模型
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ExecutionPlan)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个任务规划专家。你需要将复杂的数据查询问题分解为多个可执行的子任务。

任务分解原则：
1. 每个子任务应该是独立且明确的
2. 识别任务之间的依赖关系
3. 按照逻辑顺序排列任务
4. 每个任务应该产生中间结果供后续任务使用

任务类型：
- sql_query: 需要执行SQL查询
- calculation: 需要进行数学计算
- aggregation: 需要聚合或汇总数据

{format_instructions}
"""),
            ("user", """用户问题: {question}

数据库Schema:
{schema}

请将这个复杂问题分解为多个子任务，并制定执行计划。
""")
        ])
    
    def create_plan(self, question: str, schema: str) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            question: 用户问题
            schema: 数据库schema
            
        Returns:
            ExecutionPlan: 执行计划
        """
        chain = self.prompt | self.llm | self.parser
        
        result = chain.invoke({
            "question": question,
            "schema": schema,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def get_next_task(
        self, 
        plan: ExecutionPlan, 
        completed_tasks: List[int]
    ) -> SubTask | None:
        """
        获取下一个可执行的任务
        
        Args:
            plan: 执行计划
            completed_tasks: 已完成的任务ID列表
            
        Returns:
            下一个可执行的任务，如果没有则返回None
        """
        for task in plan.tasks:
            # 跳过已完成的任务
            if task.task_id in completed_tasks:
                continue
            
            # 检查依赖是否满足
            if all(dep_id in completed_tasks for dep_id in task.depends_on):
                return task
        
        return None
    
    def is_plan_complete(
        self, 
        plan: ExecutionPlan, 
        completed_tasks: List[int]
    ) -> bool:
        """
        检查计划是否完成
        
        Args:
            plan: 执行计划
            completed_tasks: 已完成的任务ID列表
            
        Returns:
            计划是否完成
        """
        return len(completed_tasks) == len(plan.tasks)
    
    def get_task_results_context(
        self, 
        task_results: Dict[int, Any]
    ) -> str:
        """
        获取任务结果的上下文描述
        
        Args:
            task_results: 任务结果字典 {task_id: result}
            
        Returns:
            上下文描述
        """
        if not task_results:
            return ""
        
        context = "已完成的任务结果:\n"
        for task_id, result in task_results.items():
            context += f"\n任务{task_id}的结果:\n{result}\n"
        
        return context

