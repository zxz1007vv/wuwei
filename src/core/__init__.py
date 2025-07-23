"""
围棋核心逻辑模块
包含游戏规则和特征提取
"""

from .game import Go, toDigit, toPosition, toStrPosition
from .features import getAllFeatures 