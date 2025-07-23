"""
Self-play reinforcement learning module for training Go AI.
"""

from .self_play_env import SelfPlayEnv, ReplayBuffer
from .trainer import self_play_training, main as train_main
