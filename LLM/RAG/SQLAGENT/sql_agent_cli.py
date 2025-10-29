#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 sql_agent_demo.py 的命令行对话程序
支持选择不同的 LLM 后端（vLLM、Ollama等）并执行 SQL Agent 业务
"""

import os
import sys
import traceback
from typing import Optional, Dict, Any
from sql_agent.sql_agent_demo import SQLAgentDemo
from models.ChatModel import Ollama, Bailian, VLLM, _get_ollama_llm, _get_bailian_llm, _get_vllm_llm
from llm_wrapper import LLMWrapper, Qwen3ReActInterceptor
from config import get_config


class SQLAgentCLI:
    """SQL Agent 命令行交互程序"""
    
    def __init__(self):
        self.agent: Optional[SQLAgentDemo] = None
        self.current_model: Optional[str] = None
        self.config = get_config()
        
    def print_banner(self):
        """打印程序横幅"""
        print("=" * 80)
        print("🤖 SQL Agent 命令行对话程序")
        print("=" * 80)
        print("支持多种 LLM 后端：")
        print("  • Ollama - 本地模型服务")
        print("  • vLLM - 高性能推理服务")
        print("  • Bailian - 阿里云通义千问")
        print("=" * 80)
    
    def show_model_menu(self):
        """显示模型选择菜单"""
        print("\n📋 请选择 LLM 后端：")
        print("1. Ollama (本地模型服务)")
        print("2. vLLM (高性能推理服务)")
        print("3. Bailian (阿里云通义千问)")
        print("0. 退出程序")
        print("-" * 40)
    
    def get_model_choice(self) -> str:
        """获取用户选择的模型"""
        while True:
            try:
                choice = input("请输入选择 (0-3): ").strip()
                if choice == "0":
                    return "exit"
                elif choice == "1":
                    return Ollama
                elif choice == "2":
                    return VLLM
                elif choice == "3":
                    return Bailian
                else:
                    print("❌ 无效选择，请输入 0-3 之间的数字")
            except KeyboardInterrupt:
                print("\n\n👋 程序已退出")
                sys.exit(0)
            except Exception as e:
                print(f"❌ 输入错误: {e}")
    
    def create_llm(self, model_type: str):
        """根据模型类型创建 LLM 实例"""
        print(f"\n🔄 正在初始化 {model_type} 模型...")
        
        try:
            if model_type == Ollama:
                llm = _get_ollama_llm()
                if llm is None:
                    raise ValueError("Ollama 模型初始化失败")
            elif model_type == VLLM:
                llm = _get_vllm_llm()
                if llm is None:
                    raise ValueError("vLLM 模型初始化失败")
            elif model_type == Bailian:
                llm = _get_bailian_llm()
                if llm is None:
                    raise ValueError("Bailian 模型初始化失败")
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 使用 LLM Wrapper 包装，添加拦截器
            wrapped_llm = LLMWrapper(
                llm, 
                interceptors=[Qwen3ReActInterceptor(jsonl_file="./generate_data/qwen3_react_history.jsonl")]
            )
            
            print(f"✅ {model_type} 模型初始化成功")
            return wrapped_llm
            
        except Exception as e:
            print(f"❌ {model_type} 模型初始化失败: {e}")
            print(f"📋 错误详情:\n{traceback.format_exc()}")
            return None
    
    def initialize_agent(self, model_type: str) -> bool:
        """初始化 SQL Agent"""
        print(f"\n🔄 正在初始化 SQL Agent ({model_type})...")
        
        try:
            # 创建自定义的 SQL Agent Demo
            self.agent = CustomSQLAgentDemo(model_type)
            self.current_model = model_type
            print(f"✅ SQL Agent ({model_type}) 初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ SQL Agent 初始化失败: {e}")
            print(f"📋 错误详情:\n{traceback.format_exc()}")
            return False
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 使用说明：")
        print("• 直接输入问题，Agent 会生成 SQL 查询并执行")
        print("• 输入 'help' 或 '?' 显示此帮助")
        print("• 输入 'model' 或 'm' 切换模型")
        print("• 输入 'exit' 或 'quit' 退出程序")
        print("• 输入 'clear' 清屏")
        print("• 输入 'status' 查看当前状态")
        print("\n💡 示例问题：")
        print("• 技术部有多少员工？")
        print("• 工资最高的3名员工是谁？")
        print("• 每个部门的平均工资是多少？")
        print("-" * 60)
    
    def show_status(self):
        """显示当前状态"""
        print(f"\n📊 当前状态：")
        print(f"• 模型类型: {self.current_model}")
        print(f"• 数据库: {self.config.get('database_uri', '未设置')}")
        print(f"• Agent 状态: {'已初始化' if self.agent else '未初始化'}")
        print("-" * 40)
    
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
    
    def process_query(self, question: str):
        """处理用户查询"""
        if not self.agent:
            print("❌ Agent 未初始化，请先选择模型")
            return
        
        print(f"\n🔍 正在处理查询: {question}")
        print("-" * 60)
        
        try:
            # 执行查询
            result = self.agent.query(question)
            
            print("\n✅ 查询完成")
            print("=" * 60)
            print("📋 查询结果:")
            print(result.get('output', result) if isinstance(result, dict) else result)
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 查询失败: {e}")
            print(f"📋 错误详情:\n{traceback.format_exc()}")
    
    def run(self):
        """运行主程序"""
        self.print_banner()
        
        # 初始模型选择
        self.show_model_menu()
        model_choice = self.get_model_choice()
        
        if model_choice == "exit":
            print("👋 程序已退出")
            return
        
        # 初始化 Agent
        if not self.initialize_agent(model_choice):
            print("❌ Agent 初始化失败，程序退出")
            return
        
        # 显示帮助
        self.show_help()
        
        # 主循环
        while True:
            try:
                print(f"\n[{self.current_model}] 请输入问题 (输入 'help' 查看帮助):")
                user_input = input("> ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 程序已退出")
                    break
                elif user_input.lower() in ['help', '?']:
                    self.show_help()
                elif user_input.lower() in ['model', 'm']:
                    self.show_model_menu()
                    new_model = self.get_model_choice()
                    if new_model != "exit":
                        if self.initialize_agent(new_model):
                            print(f"✅ 已切换到 {new_model} 模型")
                        else:
                            print(f"❌ 切换到 {new_model} 失败")
                elif user_input.lower() == 'clear':
                    self.clear_screen()
                elif user_input.lower() == 'status':
                    self.show_status()
                else:
                    # 处理普通查询
                    self.process_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 程序已退出")
                break
            except Exception as e:
                print(f"\n❌ 程序错误: {e}")
                print(f"📋 错误详情:\n{traceback.format_exc()}")


class CustomSQLAgentDemo(SQLAgentDemo):
    """自定义的 SQL Agent Demo，支持动态模型选择"""
    
    def __init__(self, model_type: str, database_uri: str = None):
        self.model_type = model_type
        self.database_uri = database_uri
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化组件"""
        from config import get_config
        from db_utils import LoggingSQLDatabase
        
        config = get_config()
        
        # 设置数据库URI
        if not self.database_uri:
            self.database_uri = config.get("database_uri")
        
        # 修正数据库URI格式
        self.database_uri = self._fix_sqlite_uri(self.database_uri)
        print(f"📋 使用数据库URI: {self.database_uri}")
        
        # 初始化LLM
        self.llm = self._create_llm()
        
        # 初始化数据库连接
        self.db = LoggingSQLDatabase.from_uri(self.database_uri)
        
        # 测试连接
        self._test_connections()
        
        # 创建Agent
        self.agent = self.create_agent()
    
    def _create_llm(self):
        """创建LLM实例"""
        if self.model_type == Ollama:
            llm = _get_ollama_llm()
        elif self.model_type == VLLM:
            llm = _get_vllm_llm()
        elif self.model_type == Bailian:
            llm = _get_bailian_llm()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        if llm is None:
            raise ValueError(f"{self.model_type} 模型初始化失败")
        
        # 使用 LLM Wrapper 包装
        wrapped_llm = LLMWrapper(
            llm, 
            interceptors=[Qwen3ReActInterceptor(jsonl_file="./generate_data/qwen3_react_history.jsonl")]
        )
        
        return wrapped_llm
    
    def _test_connections(self):
        """测试连接"""
        # 测试 LLM 连接
        try:
            response = self.llm.invoke("你是谁？请用一句话自我介绍。")
            print(f"✅ LLM 测试成功")
        except Exception as e:
            print(f"❌ LLM 调用失败: {str(e)}")
            raise
        
        # 测试数据库连接
        try:
            tables = self.db.get_usable_table_names()
            print(f"✅ 数据库连接成功！")
        except Exception as e:
            print(f"❌ 数据库连接失败: {str(e)}")
            raise
    
    @staticmethod
    def _fix_sqlite_uri(uri: str) -> str:
        """修正SQLite URI格式"""
        if not uri or uri.strip() == '':
            raise ValueError("数据库URI不能为空")
        
        # 如果不是sqlite URI，假设是文件路径
        if not uri.startswith('sqlite:'):
            abs_path = os.path.abspath(uri).replace('\\', '/')
            return f"sqlite:///{abs_path}"
        
        # 修正常见的格式错误
        if uri.startswith('sqlite://') and not uri.startswith('sqlite:///'):
            path = uri[9:]  # 移除 "sqlite://"
            path = path.replace('\\', '/')
            return f"sqlite:///{path}"
        
        # 统一路径分隔符
        uri = uri.replace('\\', '/')
        return uri


def main():
    """主函数"""
    try:
        cli = SQLAgentCLI()
        cli.run()
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        print(f"📋 错误详情:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
