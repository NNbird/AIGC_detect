#!/bin/bash
# save as install_pytorch.sh

echo "=== PyTorch 安装诊断与修复 ==="

# 1. 系统信息
echo -e "\n1. 系统信息:"
echo "系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"
echo "CPU架构: $(uname -m)"
echo "Python3版本: $(python3 --version 2>/dev/null || echo '未安装')"

# 2. Python 环境
echo -e "\n2. Python 环境:"
python3 -c "import sys; print(f'Python路径: {sys.executable}')"
python3 -c "import sys; print(f'Python版本: {sys.version}')"

# 3. pip 状态
echo -e "\n3. pip 状态:"
python3 -m pip --version 2>/dev/null || echo "pip 未安装"

# 4. 安装 PyTorch
echo -e "\n4. 安装 PyTorch..."
echo "正在安装，请稍候..."

# 尝试多种安装方式
install_methods=(
    "python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    "python3 -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple"
    "python3 -m pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu"
)

for method in "${install_methods[@]}"; do
    echo -e "\n尝试: $method"
    if $method; then
        echo "✓ 安装成功！"
        break
    else
        echo "✗ 安装失败，尝试下一种方法"
        sleep 2
    fi
done

# 5. 验证安装
echo -e "\n5. 验证安装:"
if python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>/dev/null; then
    echo "✓ PyTorch 安装成功！"
else
    echo "✗ PyTorch 未安装成功"
    
    # 尝试使用 conda 安装
    read -p "是否尝试使用 conda 安装？(y/n): " choice
    if [ "$choice" = "y" ]; then
        echo "安装 Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        source $HOME/miniconda/bin/activate
        conda install pytorch torchvision cpuonly -c pytorch -y
    fi
fi

echo -e "\n=== 完成 ==="