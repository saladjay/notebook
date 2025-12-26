@echo off
REM Windows安装脚本

echo ========================================
echo SQL Agent 自动安装脚本 (Windows)
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/4] 检查Python版本...
python --version

echo.
echo [2/4] 创建虚拟环境...
if exist venv (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    echo 虚拟环境创建完成
)

echo.
echo [3/4] 激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo [4/4] 配置环境变量...
if exist .env (
    echo .env 文件已存在，跳过复制
) else (
    copy env_template.txt .env
    echo .env 文件已创建，请编辑此文件填入您的配置
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步：
echo 1. 编辑 .env 文件，填入您的配置
echo 2. 运行验证脚本: python verify_installation.py
echo 3. 启动程序: python main.py
echo.

pause





