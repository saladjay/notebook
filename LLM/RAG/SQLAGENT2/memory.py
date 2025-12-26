"""
记忆系统模块
用于存储和检索对话历史、执行历史等信息
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory


class MemorySystem:
    """记忆系统"""
    
    def __init__(self, llm=None):
        """
        初始化记忆系统
        
        Args:
            llm: 语言模型，用于对话摘要
        """
        # 对话记忆
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        
        # SQL查询历史
        self.sql_history: List[Dict[str, Any]] = []
        
        # 知识记忆（用户反馈、偏好等）
        self.knowledge_memory: Dict[str, Any] = {}
    
    def add_conversation(self, user_input: str, ai_response: str):
        """添加对话记录"""
        self.conversation_memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
    
    def add_execution(self, task: str, result: Any, metadata: Optional[Dict] = None):
        """添加执行记录"""
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result,
            "metadata": metadata or {}
        }
        self.execution_history.append(execution_record)
    
    def add_sql_query(self, question: str, sql: str, result: Any, success: bool):
        """添加SQL查询记录"""
        sql_record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "sql": sql,
            "result": result,
            "success": success
        }
        self.sql_history.append(sql_record)
    
    def get_recent_conversations(self, n: int = 5) -> List[Dict[str, str]]:
        """获取最近的对话记录"""
        messages = self.conversation_memory.chat_memory.messages
        recent_messages = messages[-n*2:] if len(messages) > n*2 else messages
        
        conversations = []
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                conversations.append({
                    "user": recent_messages[i].content,
                    "assistant": recent_messages[i + 1].content
                })
        return conversations
    
    def get_recent_sql_queries(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的SQL查询"""
        return self.sql_history[-n:]
    
    def get_successful_sql_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近成功的SQL查询"""
        successful_queries = [q for q in self.sql_history if q["success"]]
        return successful_queries[-n:]
    
    def save_knowledge(self, key: str, value: Any):
        """保存知识"""
        self.knowledge_memory[key] = value
    
    def get_knowledge(self, key: str, default=None) -> Any:
        """获取知识"""
        return self.knowledge_memory.get(key, default)
    
    def clear(self):
        """清空所有记忆"""
        self.conversation_memory.clear()
        self.execution_history.clear()
        self.sql_history.clear()
        self.knowledge_memory.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return {
            "total_conversations": len(self.conversation_memory.chat_memory.messages) // 2,
            "total_executions": len(self.execution_history),
            "total_sql_queries": len(self.sql_history),
            "successful_sql_queries": len([q for q in self.sql_history if q["success"]]),
            "knowledge_items": len(self.knowledge_memory)
        }


