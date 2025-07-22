#!/usr/bin/env python3
"""
Irene - 围棋AI主程序
使用方法:
    python main.py gtp                  # 启动GTP协议服务
    python main.py gtp MCTS             # 启动GTP协议服务(MCTS模式)
    python main.py train policy         # 训练策略网络
    python main.py train playout        # 训练快速策略网络
    python main.py train value          # 训练价值网络
    python main.py prepare_data         # 准备训练数据
    python main.py filter_sgf           # 过滤SGF文件
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Irene - 围棋AI主程序")
    cmd = parser.add_subparsers(dest='command', help='可用命令', required=True)
    gtp = cmd.add_parser('gtp', help='启动GTP协议服务')
    gtp.add_argument('mode', nargs='?', default='PolicyNet', choices=[
                     'PolicyNet', 'MCTS'], help='GTP模式，默认为PolicyNet，MCTS为蒙特卡洛树搜索模式')
    train = cmd.add_parser('train', help='训练网络')
    train.add_argument('network_type', choices=['policy', 'playout', 'value'],
                       help='指定要训练的网络类型: policy(策略网络), playout(快速策略网络), value(价值网络)')
    filter_sgf = cmd.add_parser('filter_sgf', help='过滤SGF文件')
    prepare_data = cmd.add_parser('prepare_data', help='准备训练数据')

    args = parser.parse_args()

    # 根据命令行参数执行相应的功能
    if args.command == 'gtp':
        from src.interface.gtp import main as gtp_main
        gtp_main(args.mode)

    elif args.command == 'train':
        from src.training.trainer import main as train_main
        train_main(args.network_type)

    elif args.command == 'prepare_data':
        from src.data.prepare import main as prepare_main
        prepare_main()

    elif args.command == 'filter_sgf':
        from src.data.filter import main as filter_main
        filter_main()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
