"""
澄清器模块
当问题不明确时，与用户交互以获取更多信息
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class Clarifier:
    """澄清器"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化澄清器
        
        Args:
            llm: 语言模型
        """
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL Agent的澄清模块。当用户的问题不够明确时，你需要：
1. 识别问题中的模糊或缺失的信息
2. 生成友好的澄清问题
3. 提供可能的选项（如果适用）

要求：
- 一次只问一个最关键的问题
- 语言要友好、自然
- 如果可能，提供选项让用户选择
- 解释为什么需要这个信息
"""),
            ("user", """用户问题: {question}

数据库schema: {schema}

分类理由: {reasoning}

请生成一个澄清问题，帮助明确用户的需求。
""")
        ])
    
    def generate_clarification(
        self, 
        question: str, 
        schema: str,
        reasoning: str
    ) -> str:
        """
        生成澄清问题
        
        Args:
            question: 原始用户问题
            schema: 数据库schema
            reasoning: 需要澄清的理由
            
        Returns:
            澄清问题文本
        """
        chain = self.prompt | self.llm
        
        response = chain.invoke({
            "question": question,
            "schema": schema,
            "reasoning": reasoning
        })
        
        return response.content
    
    def process_clarification_response(
        self,
        original_question: str,
        clarification: str,
        user_response: str
    ) -> str:
        """
        处理用户的澄清回答，生成完整的问题
        
        Args:
            original_question: 原始问题
            clarification: 澄清问题
            user_response: 用户的回答
            
        Returns:
            完整的问题描述
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "将原始问题、澄清问题和用户回答组合成一个完整明确的问题。只返回组合后的问题，不要其他内容。"),
            ("user", """原始问题: {original_question}
澄清问题: {clarification}
用户回答: {user_response}

请生成完整的问题:""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "original_question": original_question,
            "clarification": clarification,
            "user_response": user_response
        })
        
        return response.content.strip()


