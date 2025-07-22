#!/usr/bin/env python3
"""
Irene - 围棋AI主程序
使用方法:
    python main.py gtp                  # 启动GTP协议服务
    python main.py gtp MCTS             # 启动GTP协议服务(MCTS模式)
    python main.py train policy         # 训练策略网络
    python main.py train playout        # 训练快速策略网络
    python main.py train value          # 训练价值网络
    python main.py prepare-data         # 准备训练数据
    python main.py filter-sgf           # 过滤SGF文件
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == 'gtp':
        from src.interface.gtp import main as gtp_main
        gtp_main()
    
    elif command == 'train':
        if len(sys.argv) < 3:
            print("请指定训练的网络类型: policy, playout, value")
            return
        
        network_type = sys.argv[2]
        from src.training.trainer import main as train_main
        train_main(network_type)
    
    elif command == 'prepare-data':
        from src.data.prepare import main as prepare_main
        prepare_main()
    
    elif command == 'filter-sgf':
        from src.data.filter import main as filter_main
        filter_main()
    
    else:
        print(f"未知命令: {command}")
        print(__doc__)

if __name__ == '__main__':
    main() 