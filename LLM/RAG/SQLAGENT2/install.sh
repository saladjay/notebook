#!/bin/bash
# Linux/Mac安装脚本

echo "========================================"
echo "SQL Agent 自动安装脚本 (Linux/Mac)"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

echo "[1/4] 检查Python版本..."
python3 --version

echo ""
echo "[2/4] 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo "虚拟环境创建完成"
fi

echo ""
echo "[3/4] 激活虚拟环境并安装依赖..."
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "[4/4] 配置环境变量..."
if [ -f ".env" ]; then
    echo ".env 文件已存在，跳过复制"
else
    cp env_template.txt .env
    echo ".env 文件已创建，请编辑此文件填入您的配置"
fi

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "下一步："
echo "1. 编辑 .env 文件，填入您的配置: nano .env"
echo "2. 运行验证脚本: python verify_installation.py"
echo "3. 启动程序: python main.py"
echo ""

