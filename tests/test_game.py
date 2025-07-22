"""
围棋游戏逻辑测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.game import Go, toDigit, toPosition

def test_basic_moves():
    """测试基本移动"""
    go = Go()
    
    # 测试正常落子
    assert go.move(1, 4, 4) == True
    assert go.board[4, 4] == 1
    
    # 测试重复落子
    assert go.move(-1, 4, 4) == False
    
    print("基本移动测试通过")

def test_capture():
    """测试吃子"""
    go = Go()
    
    # 设置一个可以吃子的局面
    go.move(1, 15, 4)
    go.move(-1, 15, 5)
    go.move(1, 16, 3)
    go.move(-1, 16, 4)
    go.move(1, 17, 4)
    go.move(-1, 17, 5)
    go.move(1, 16, 5)
    go.move(-1, 16, 6)
    
    # 检查是否正确吃子
    assert go.board[16, 4] == 0
    
    print("吃子测试通过")

def test_liberty():
    """测试气的计算"""
    go = Go()
    
    go.move(1, 4, 4)
    go.move(1, 4, 5)
    go.move(1, 4, 6)
    go.move(1, 5, 4)
    go.move(-1, 5, 5)
    
    # 检查气数
    assert go.liberty[4, 4] == go.liberty[4, 5] == go.liberty[4, 6] == go.liberty[5, 4] == 8
    assert go.liberty[5, 5] == 2
    
    print("气数计算测试通过")

def test_coordinate_conversion():
    """测试坐标转换"""
    # 测试数字到坐标的转换
    x, y = toPosition(0)
    assert x == 0 and y == 0
    
    x, y = toPosition(361)
    assert x is None and y is None
    
    # 测试坐标到数字的转换
    digit = toDigit(0, 0)
    assert digit == 0
    
    print("坐标转换测试通过")

if __name__ == '__main__':
    test_basic_moves()
    test_capture()
    test_liberty()
    test_coordinate_conversion()
    print("所有测试通过！") 