#!/usr/bin/env python3
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="RL Self-Play Training Module")
    parser.add_argument('--models-dir', type=str, default='models',
                        help='预训练模型目录')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数，仅对自我对弈训练有效')
    parser.add_argument('--games', type=int, default=100,
                        help='每轮训练的对局数，仅对自我对弈训练有效')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='训练批次大小，仅对自我对弈训练有效')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='保存检查点的间隔轮数，仅对自我对弈训练有效')
    parser.add_argument('--save-games', action='store_true',
                        help='是否保存对局记录，仅对自我对弈训练有效')
    parser.add_argument('--policy-only', action='store_true',
                        help='仅使用策略网络进行训练，不使用MCTS')

    args = parser.parse_args()
    import torch
    from self_play.trainer import self_play_training
    self_play_training(
        models_dir=args.models_dir,
        num_epochs=args.epochs,
        games_per_epoch=args.games,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        save_games=args.save_games,
        policy_only=args.policy_only,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


if __name__ == '__main__':
    main()
