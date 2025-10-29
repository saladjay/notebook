@echo off
chcp 65001 >nul
echo ========================================
echo SQL Agent Demo 启动器
echo ========================================
echo.

:menu
echo 请选择要执行的操作:
echo.
echo [1] 安装依赖包
echo [2] 初始化数据库
echo [3] 环境检查
echo [4] 运行 SQL Agent (完整版)
echo [5] 运行 SQL Chain (简化版)
echo [6] 运行性能优化版 (推荐生产) ⚡
echo [7] 查看当前配置
echo [0] 退出
echo.
set /p choice=请输入选项 (0-7): 

if "%choice%"=="1" goto install
if "%choice%"=="2" goto initdb
if "%choice%"=="3" goto check
if "%choice%"=="4" goto agent
if "%choice%"=="5" goto chain
if "%choice%"=="6" goto optimized
if "%choice%"=="7" goto showconfig
if "%choice%"=="0" goto end
goto menu

:install
echo.
echo 正在安装依赖包...
pip install -r requirements.txt
echo.
pause
goto menu

:initdb
echo.
echo 正在初始化数据库...
python init_database.py
echo.
pause
goto menu

:check
echo.
echo 正在检查环境...
python quick_test.py
echo.
pause
goto menu

:agent
echo.
echo 正在启动 SQL Agent...
python sql_agent_demo.py
echo.
pause
goto menu

:chain
echo.
echo 正在启动 SQL Chain...
python sql_chain_demo.py
echo.
pause
goto menu

:optimized
echo.
echo 正在启动性能优化版...
python sql_agent_optimized.py
echo.
pause
goto menu

:showconfig
echo.
echo 正在显示配置...
python config.py
echo.
pause
goto menu

:end
echo.
echo 再见！
exit /b 0

