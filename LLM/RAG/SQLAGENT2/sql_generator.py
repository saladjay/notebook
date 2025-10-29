"""
SQL生成器模块
将自然语言问题转换为SQL查询
"""
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class SQLQueryResult(BaseModel):
    """SQL查询生成结果"""
    sql: str = Field(description="生成的SQL查询语句")
    explanation: str = Field(description="SQL查询的解释说明")
    confidence: float = Field(description="生成置信度(0-1)", ge=0.0, le=1.0)


class SQLGenerator:
    """SQL生成器"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化SQL生成器
        
        Args:
            llm: 语言模型
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=SQLQueryResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的SQL查询生成器。根据用户的自然语言问题和数据库schema，生成准确的SQL查询。

要求：
1. SQL必须符合标准SQL语法
2. 只生成SELECT查询，不要生成修改数据的语句
3. 考虑性能优化，合理使用索引
4. 使用适当的JOIN、WHERE、GROUP BY等子句
5. 为查询添加必要的注释

{format_instructions}
"""),
            ("user", """用户问题: {question}

数据库Schema:
{schema}

{context}

请生成SQL查询。""")
        ])
    
    def generate_sql(
        self, 
        question: str,
        schema: str,
        context: str = "",
        examples: Optional[list] = None
    ) -> SQLQueryResult:
        """
        生成SQL查询
        
        Args:
            question: 用户问题
            schema: 数据库schema
            context: 额外的上下文信息（业务规则、查询模式等）
            examples: 示例查询（few-shot learning）
            
        Returns:
            SQLQueryResult: SQL查询结果
        """
        # 构建上下文
        full_context = context
        if examples:
            full_context += "\n\n示例查询:\n"
            for i, example in enumerate(examples, 1):
                full_context += f"\n示例{i}:\n"
                full_context += f"问题: {example['question']}\n"
                full_context += f"SQL: {example['sql']}\n"
        
        chain = self.prompt | self.llm | self.parser
        
        result = chain.invoke({
            "question": question,
            "schema": schema,
            "context": full_context,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result
    
    def refine_sql(
        self,
        original_sql: str,
        error_message: str,
        schema: str
    ) -> SQLQueryResult:
        """
        修正SQL查询
        
        Args:
            original_sql: 原始SQL
            error_message: 错误信息
            schema: 数据库schema
            
        Returns:
            SQLQueryResult: 修正后的SQL
        """
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL查询修正专家。根据错误信息修正SQL查询。

{format_instructions}
"""),
            ("user", """原始SQL:
{original_sql}

错误信息:
{error_message}

数据库Schema:
{schema}

请分析错误并生成修正后的SQL。""")
        ])
        
        chain = refine_prompt | self.llm | self.parser
        
        result = chain.invoke({
            "original_sql": original_sql,
            "error_message": error_message,
            "schema": schema,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return result

