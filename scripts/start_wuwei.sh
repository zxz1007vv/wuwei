#!/bin/bash
# wuwei围棋AI启动脚本
# 使用方法：
# ./start_wuwei.sh          - 使用策略网络模式
# ./start_wuwei.sh MCTS     - 使用MCTS搜索模式

cd /home/zxz/wuwei
source /home/zxz/anaconda3/etc/profile.d/conda.sh
conda activate go

if [ "$1" = "MCTS" ]; then
    echo "启动wuwei  (MCTS模式)..." >&2
    python main.py gtp MCTS
else
    echo "启动wuwei (策略网络模式)..." >&2
    python main.py gtp
fi 