from setuptools import setup, find_packages
'''
setup(
    name="your_package_name",
    version="0.1.0",
    packages=find_packages(),  # 自动找到所有包
    
    # 包含非Python文件
    package_data={
        # 如果在your_package目录下有数据文件:
        "your_package": ["data/*.json", "data/*.txt"],
    },
    
    # 包含任意位置的数据文件
    data_files=[
        ('json', ['json/*.json']),  # (目标目录, 源文件列表)
        ('image', ['image/*.jpg', 'image/*.png']),
    ],
    
    # 如果你有一些脚本需要安装到bin目录
    scripts=['scripts/script1.py', 'scripts/script2.py'],
    
    # 或者使用entry_points定义可执行命令
    entry_points={
        'console_scripts': [
            'your-command=your_package.module:main_function',
        ],
    },
)

主要说明：
packages=find_packages() - 自动查找所有包含 __init__.py 的目录
package_data - 包含包内的非Python文件
data_files - 指定要复制到系统的任意位置的文件
scripts - 安装可执行脚本
entry_points - 定义命令行入口点
如果你想要更精确的控制包含哪些包，可以手动指定：

setup(
    # ...
    packages=['your_package', 'your_package.subpackage'],
    # ...
)

setup(
    # ...
    packages=find_packages(
        include=['your_package*'],  # 包含的包
        exclude=['your_package.tests*']  # 排除的包
    ),
    # ...
)
'''
setup(
    name='DeepLearningCommonPythonLib',
    version='0.1',
    packages=find_packages(),
    author='dengyj',
    author_email='dyj394041771@sina.com',
    description='Common Python Lib for Deep Learning',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
        # 防止生成 .pyc 文件
    options={
        'py2exe': {'bundle_files': 1},
        'bdist_wheel': {'universal': True},
    },
    # 确保MANIFEST.in文件被包含
    include_package_data=True,
)