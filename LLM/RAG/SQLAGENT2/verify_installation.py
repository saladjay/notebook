"""
安装验证脚本
检查环境配置是否正确
"""
import sys
import os
from pathlib import Path


def print_header(text):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def check_python_version():
    """检查Python版本"""
    print("\n检查Python版本...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro} (满足要求 >= 3.8)")
        return True
    else:
        print(f"✗ Python版本过低: {version.major}.{version.minor}.{version.micro} (需要 >= 3.8)")
        return False


def check_dependencies():
    """检查依赖包"""
    print("\n检查依赖包...")
    
    required_packages = {
        'langchain': 'LangChain框架',
        'langchain_openai': 'OpenAI集成',
        'openai': 'OpenAI API',
        'sqlalchemy': '数据库连接',
        'pandas': '数据处理',
        'pydantic': '数据验证',
        'dotenv': '环境变量'
    }
    
    all_installed = True
    for package, description in required_packages.items():
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"✓ {package:20s} - {description}")
        except ImportError:
            print(f"✗ {package:20s} - {description} (未安装)")
            all_installed = False
    
    return all_installed


def check_env_file():
    """检查环境变量文件"""
    print("\n检查环境变量配置...")
    
    env_path = Path(".env")
    if not env_path.exists():
        print("✗ .env 文件不存在")
        print("  请复制 env_template.txt 为 .env 并填入配置")
        return False
    
    print("✓ .env 文件存在")
    
    # 读取并检查关键配置
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API密钥',
        'DATABASE_URI': '数据库连接URI'
    }
    
    all_configured = True
    for var, description in required_vars.items():
        if var in content:
            # 检查是否为默认值
            if 'your_api_key_here' in content or \
               'user:password@localhost' in content:
                print(f"⚠ {var} - {description} (使用默认值，请修改)")
                all_configured = False
            else:
                print(f"✓ {var} - {description}")
        else:
            print(f"✗ {var} - {description} (未配置)")
            all_configured = False
    
    return all_configured


def check_database_connection():
    """检查数据库连接"""
    print("\n检查数据库连接...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        database_uri = os.getenv('DATABASE_URI')
        if not database_uri:
            print("✗ DATABASE_URI 未配置")
            return False
        
        if 'your' in database_uri.lower() or 'password' in database_uri:
            print("⚠ 数据库URI看起来是默认值，跳过连接测试")
            return None
        
        from sqlalchemy import create_engine
        engine = create_engine(database_uri)
        
        with engine.connect() as conn:
            print(f"✓ 数据库连接成功: {database_uri.split('@')[1] if '@' in database_uri else 'local'}")
            return True
            
    except Exception as e:
        print(f"✗ 数据库连接失败: {str(e)}")
        return False


def check_openai_api():
    """检查OpenAI API"""
    print("\n检查OpenAI API...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("✗ OPENAI_API_KEY 未配置")
            return False
        
        if 'your' in api_key.lower():
            print("⚠ API密钥看起来是默认值，跳过API测试")
            return None
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # 尝试一个简单的API调用
        response = client.models.list()
        print(f"✓ OpenAI API连接成功")
        return True
            
    except Exception as e:
        print(f"✗ OpenAI API连接失败: {str(e)}")
        return False


def check_module_imports():
    """检查模块导入"""
    print("\n检查项目模块...")
    
    modules = [
        'agent',
        'intent_router',
        'sql_generator',
        'query_executor',
        'planner',
        'clarifier',
        'knowledge_base',
        'security',
        'memory',
        'config'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py")
        except Exception as e:
            print(f"✗ {module}.py - 导入失败: {str(e)}")
            all_ok = False
    
    return all_ok


def print_summary(results):
    """打印总结"""
    print_header("验证总结")
    
    all_passed = all(r for r in results.values() if r is not None)
    
    for check, result in results.items():
        if result is True:
            status = "✓ 通过"
        elif result is False:
            status = "✗ 失败"
        else:
            status = "⚠ 跳过"
        print(f"{status:10s} - {check}")
    
    print("\n" + "="*60)
    
    if all_passed and None not in results.values():
        print("\n🎉 所有检查通过！可以开始使用 SQL Agent。")
        print("\n运行: python main.py")
    else:
        print("\n⚠️  部分检查未通过，请根据上述提示进行修复。")
        print("\n常见问题：")
        print("1. 依赖未安装: pip install -r requirements.txt")
        print("2. .env未配置: 复制 env_template.txt 为 .env 并填入配置")
        print("3. 数据库连接失败: 检查数据库URI和网络连接")
        print("4. API密钥无效: 检查OpenAI API密钥")


def main():
    """主函数"""
    print_header("SQL Agent 安装验证")
    
    print("""
本脚本将检查以下项目：
  1. Python版本
  2. 依赖包安装
  3. 环境变量配置
  4. 数据库连接
  5. OpenAI API连接
  6. 项目模块导入
    """)
    
    results = {
        'Python版本': check_python_version(),
        '依赖包': check_dependencies(),
        '环境变量': check_env_file(),
        '数据库连接': check_database_connection(),
        'OpenAI API': check_openai_api(),
        '项目模块': check_module_imports()
    }
    
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n检查已取消")
    except Exception as e:
        print(f"\n\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()





