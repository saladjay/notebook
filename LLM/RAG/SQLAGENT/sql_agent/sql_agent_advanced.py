"""
高级SQL Agent - 使用配置文件，支持更多功能
包括：日志记录、SQL验证、缓存等
"""
import logging
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType

# 导入配置
from config import get_config, COLORS

class SQLAgentAdvanced:
    """高级SQL Agent类"""
    
    def __init__(self, config=None):
        """初始化Agent"""
        self.config = config or get_config()
        self._setup_logging()
        self._init_components()
    
    def _setup_logging(self):
        """设置日志"""
        if self.config['enable_logging']:
            logging.basicConfig(
                filename=self.config['log_file'],
                level=getattr(logging, self.config['log_level']),
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info("=" * 60)
            self.logger.info("SQL Agent 启动")
            self.logger.info("=" * 60)
            self.logger.pro
        else:
            self.logger = None
    
    def _init_components(self):
        """初始化组件"""
        print(COLORS['header'])
        print("🚀 高级 SQL Agent - 初始化中...")
        print(COLORS['header'])
        
        # 初始化LLM
        print(f"\n{COLORS['info']} 初始化Ollama模型: {self.config['ollama_model']}")
        llm_kwargs = {
            'model': self.config['ollama_model'],
            'temperature': self.config['temperature'],
        }
        if self.config['ollama_base_url']:
            llm_kwargs['base_url'] = self.config['ollama_base_url']
        
        self.llm = Ollama(**llm_kwargs)
        print(f"{COLORS['success']} Ollama连接成功")
        
        if self.logger:
            self.logger.info(f"使用模型: {self.config['ollama_model']}")
        
        # 初始化数据库
        print(f"\n{COLORS['info']} 连接数据库: {self.config['database_uri']}")
        self.db = SQLDatabase.from_uri(self.config['database_uri'])
        print(f"{COLORS['success']} 数据库连接成功")
        print(f"   表: {', '.join(self.db.get_usable_table_names())}")
        
        if self.logger:
            self.logger.info(f"数据库: {self.config['database_uri']}")
            self.logger.info(f"表: {self.db.get_usable_table_names()}")
        
        # 创建Agent
        print(f"\n{COLORS['info']} 创建SQL Agent...")
        self.agent = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.config['verbose'],
            handle_parsing_errors=self.config['handle_parsing_errors'],
            max_iterations=self.config['max_iterations'],
        )
        print(f"{COLORS['success']} Agent创建成功\n")
    
    def validate_sql(self, sql: str) -> tuple[bool, str]:
        """
        验证SQL查询是否安全
        
        Returns:
            (is_valid, message)
        """
        if not self.config['enable_sql_validation']:
            return True, "验证已禁用"
        
        sql_upper = sql.upper()
        for keyword in self.config['forbidden_sql_keywords']:
            if keyword in sql_upper:
                return False, f"检测到禁止的SQL关键字: {keyword}"
        
        return True, "SQL验证通过"
    
    def query(self, question: str, show_process: bool = True) -> dict:
        """
        执行查询
        
        Args:
            question: 自然语言问题
            show_process: 是否显示处理过程
            
        Returns:
            包含结果和元数据的字典
        """
        start_time = datetime.now()
        
        if show_process:
            print(COLORS['header'])
            print(f"{COLORS['question']} 问题: {question}")
            print(COLORS['header'])
        
        if self.logger:
            self.logger.info(f"查询: {question}")
        
        try:
            # 执行查询
            result = self.agent.invoke({"input": question})
            
            # 处理结果
            output = result.get('output', '')
            
            # 计算耗时
            duration = (datetime.now() - start_time).total_seconds()
            
            if show_process:
                print(f"\n{COLORS['answer']} 答案: {output}")
                print(f"\n⏱️  耗时: {duration:.2f}秒")
            
            if self.logger:
                self.logger.info(f"答案: {output}")
                self.logger.info(f"耗时: {duration:.2f}秒")
            
            return {
                'success': True,
                'question': question,
                'answer': output,
                'duration': duration,
                'error': None,
            }
            
        except Exception as e:
            error_msg = str(e)
            duration = (datetime.now() - start_time).total_seconds()
            
            if show_process:
                print(f"\n{COLORS['error']} 错误: {error_msg}")
            
            if self.logger:
                self.logger.error(f"查询失败: {error_msg}")
            
            return {
                'success': False,
                'question': question,
                'answer': None,
                'duration': duration,
                'error': error_msg,
            }
    
    def run_examples(self):
        """运行示例查询"""
        print("\n" + COLORS['header'])
        print("📝 运行示例查询")
        print(COLORS['header'])
        
        queries = self.config['example_queries']
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'=' * 60}")
            print(f"示例 {i}/{len(queries)}")
            result = self.query(query)
            results.append(result)
            print()
        
        # 统计
        success_count = sum(1 for r in results if r['success'])
        total_time = sum(r['duration'] for r in results)
        
        print(COLORS['header'])
        print("📊 统计信息")
        print(COLORS['header'])
        print(f"  总查询数: {len(results)}")
        print(f"  成功: {success_count}")
        print(f"  失败: {len(results) - success_count}")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  平均耗时: {total_time/len(results):.2f}秒")
        print()
    
    def interactive_mode(self):
        """交互模式"""
        print(COLORS['header'])
        print("💬 进入交互模式")
        print("   输入 'quit' 或 'exit' 退出")
        print("   输入 'help' 查看帮助")
        print("   输入 'examples' 查看示例问题")
        print(COLORS['header'])
        
        while True:
            try:
                user_input = input(f"\n{COLORS['question']} 请输入问题: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 再见！")
                    break
                
                if user_input.lower() in ['help', '帮助', 'h']:
                    self._show_help()
                    continue
                
                if user_input.lower() in ['examples', '示例', 'e']:
                    self._show_examples()
                    continue
                
                if user_input.lower() in ['config', '配置', 'c']:
                    self._show_config()
                    continue
                
                # 执行查询
                self.query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n{COLORS['error']} 错误: {str(e)}")
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n" + COLORS['subheader'])
        print("📖 帮助信息")
        print(COLORS['subheader'])
        print("可用命令:")
        print("  help / h      - 显示此帮助")
        print("  examples / e  - 显示示例问题")
        print("  config / c    - 显示当前配置")
        print("  quit / q      - 退出程序")
        print("\n支持的查询类型:")
        print("  - 统计查询: '有多少员工？'")
        print("  - 筛选查询: '工资超过15000的员工'")
        print("  - 排序查询: '按工资排序的前10名员工'")
        print("  - 聚合查询: '每个部门的平均工资'")
    
    def _show_examples(self):
        """显示示例问题"""
        print("\n" + COLORS['subheader'])
        print("💡 示例问题")
        print(COLORS['subheader'])
        for i, example in enumerate(self.config['example_queries'], 1):
            print(f"  {i}. {example}")
    
    def _show_config(self):
        """显示当前配置"""
        print("\n" + COLORS['subheader'])
        print("⚙️  当前配置")
        print(COLORS['subheader'])
        print(f"  模型: {self.config['ollama_model']}")
        print(f"  数据库: {self.config['database_uri']}")
        print(f"  详细模式: {self.config['verbose']}")
        print(f"  最大迭代: {self.config['max_iterations']}")
        print(f"  日志: {self.config['enable_logging']}")


def main():
    """主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║            🚀 高级 SQL Agent                             ║
    ║            LangChain + Ollama + SQLite                   ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 创建Agent
    agent = SQLAgentAdvanced()
    
    # 询问用户选择
    print("\n请选择模式:")
    print("  [1] 运行示例查询")
    print("  [2] 交互模式")
    print("  [3] 两者都运行")
    
    choice = input("\n请输入选项 (1-3，默认3): ").strip() or "3"
    
    if choice in ['1', '3']:
        agent.run_examples()
    
    if choice in ['2', '3']:
        if choice == '3':
            input("\n按Enter键进入交互模式...")
        agent.interactive_mode()
    
    print("\n" + COLORS['header'])
    print("感谢使用！")
    print(COLORS['header'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 严重错误: {str(e)}")
        import traceback
        traceback.print_exc()

