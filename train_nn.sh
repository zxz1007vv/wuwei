#!/bin/bash

# 设置严格的错误处理
set -e  # 任何命令返回非零状态码时立即退出
set -u  # 使用未定义变量时立即退出
set -o pipefail  # 管道中任何命令失败时立即退出

echo "开始执行围棋神经网络训练脚本..."

# 错误处理函数
error_exit() {
    echo "❌ 错误: $1" >&2
    echo "脚本执行失败，已中止。" >&2
    exit 1
}

# 成功提示函数
success_msg() {
    echo "✅ $1"
}



# 激活conda环境
echo "🐍 激活conda环境..."
source /home/zxz/anaconda3/etc/profile.d/conda.sh || error_exit "无法加载conda环境配置"
conda activate go || error_exit "无法激活conda环境 'go'"
success_msg "已激活conda环境: go"

# 检查数据文件是否存在
echo "🔍 检查数据文件..."
data_exists=false

if [ -f "games.7z" ] || [ -d "games" ] || find . -name "*.sgf" -quit 2>/dev/null; then
    echo "📦 发现已存在的数据文件，跳过下载步骤"
    data_exists=true
else
    echo "📥 未发现数据文件，开始下载..."
    
    # 下载数据文件
    echo "正在下载游戏数据..."
    if ! wget https://homepages.cwi.nl/~aeb/go/games/games.7z; then
        error_exit "下载失败，请检查网络连接"
    fi
    success_msg "下载完成"
fi

# 检查并解压数据文件
if [ -f "games.7z" ] && [ ! -d "games" ]; then
    echo "📦 正在解压数据文件..."
    if ! 7z x games.7z; then
        error_exit "解压失败，请检查7z工具是否安装"
    fi
    success_msg "解压完成"
elif [ -d "games" ]; then
    success_msg "数据文件已解压"
else
    error_exit "找不到可用的数据文件"
fi

# 检查是否有SGF文件
if ! find games -name "*.sgf" -quit 2>/dev/null; then
    error_exit "在games目录中未找到SGF文件"
fi

# 创建models目录（如果不存在）
echo "📁 确保models目录存在..."
mkdir -p models || error_exit "无法创建models目录"

# 处理数据
echo "🔄 开始处理数据..."

echo "正在过滤SGF文件..."
if ! python main.py filter_sgf; then
    error_exit "SGF文件过滤失败"
fi
success_msg "SGF文件过滤完成"

echo "正在准备训练数据..."
if ! python main.py prepare_data; then
    error_exit "训练数据准备失败"
fi
success_msg "训练数据准备完成"

# 检查训练数据是否生成成功
echo "🔍 检查训练数据文件..."
required_files=("models/policyData.pt" "models/valueData.pt")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        error_exit "缺少必要的训练数据文件: $file"
    fi
done
success_msg "所有训练数据文件已准备就绪"

# 训练网络
echo "🧠 开始训练神经网络..."

echo "正在训练策略网络..."
if ! python main.py train policy; then
    error_exit "策略网络训练失败"
fi
success_msg "策略网络训练完成"

echo "正在训练快速策略网络..."
if ! python main.py train playout; then
    error_exit "快速策略网络训练失败"
fi
success_msg "快速策略网络训练完成"

echo "正在训练价值网络..."
if ! python main.py train value; then
    error_exit "价值网络训练失败"
fi
success_msg "价值网络训练完成"

echo "🎉 训练完成！所有网络已成功训练并保存到models目录"
echo "训练结果文件:"
ls -la models/*.pt 2>/dev/null || echo "注意: models目录中未找到.pt文件" 