"""
配置文件
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # OpenAI配置
    openai_api_key: str
    openai_api_base: Optional[str] = None
    model_name: str = "gpt-4"
    temperature: float = 0.0
    
    # 数据库配置
    database_uri: str
    
    # Agent配置
    max_iterations: int = 5
    verbose: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()

