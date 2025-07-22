import torch
import numpy as np
import sys
import os
from src.ai.networks import PolicyNetwork, PlayoutNetwork, ValueNetwork
from src.core.game import Go, toPosition
from src.core.features import getAllFeatures
from src.ai.mcts import MCTSNode, MCTS

# 设置随机种子
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# 获取程序路径
programPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/'

# 加载预训练模型
policyNet = PolicyNetwork()
policyNet.load_state_dict(torch.load(programPath + 'models/policyNet.pt'))

playoutNet = PlayoutNetwork()
playoutNet.load_state_dict(torch.load(programPath + 'models/playoutNet.pt'))

valueNet = ValueNetwork()
valueNet.load_state_dict(torch.load(programPath + 'models/valueNet.pt'))

# 颜色和坐标转换
colorCharToIndex = {'B': 1, 'W': -1, 'b': 1, 'w': -1}
indexToColorChar = {1: 'B', -1: 'W'}
indexToChar = []
charToIndex = {}
char = ord('A')

for i in range(19):
    indexToChar.append(chr(char))
    charToIndex[chr(char)] = i
    char += 1
    if char == ord('I'):
        char += 1


def toStrPosition(x, y):
    """将坐标转换为字符串表示"""
    if (x, y) == (None, None):
        return ''
    x = 19 - x
    y = indexToChar[y]
    return f'{y}{x}'


def getPolicyNetResult(go, willPlayColor):
    """获取策略网络的预测结果"""
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = policyNet(inputData)[0]
    return predict


def getPlayoutNetResult(go, willPlayColor):
    """获取快速策略网络的预测结果"""
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = playoutNet(inputData)[0]
    return predict


def getValueNetResult(go, willPlayColor):
    """获取价值网络的预测结果"""
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    value = valueNet(inputData)[0].item()
    return value


def getValueResult(go, willPlayColor):
    """获取简单的价值评估（基于棋子数量差）"""
    countThisColor = np.sum(go.board == willPlayColor)
    countAnotherColor = np.sum(go.board == -willPlayColor)
    return countThisColor - countAnotherColor


def genMovePolicy(go, willPlayColor):
    """使用策略网络生成移动"""
    predict = getPolicyNetResult(go, willPlayColor)
    predictReverseSortIndex = reversed(torch.argsort(predict))

    # stderr 输出 valueNet 结果
    value = getValueResult(go, willPlayColor)
    sys.stderr.write(f'{willPlayColor} {value}\n')

    for predictIndex in predictReverseSortIndex:
        x, y = toPosition(predictIndex)
        if (x, y) == (None, None):
            print('pass')
            return
        moveResult = go.move(willPlayColor, x, y)
        strPosition = toStrPosition(x, y)

        if moveResult == False:
            sys.stderr.write(f'Illegal move: {strPosition}\n')
        else:
            print(strPosition)
            break


def genMoveMCTS(go, willPlayColor, debug=False):
    """使用MCTS生成移动"""
    root = MCTSNode(go, willPlayColor, None)

    bestNextNode = MCTS(root, getPolicyNetResult, getPlayoutNetResult, getValueNetResult, debug=debug)
    
    # 如果MCTS搜索失败，回退到策略网络
    if bestNextNode is None:
        sys.stderr.write('MCTS搜索失败：未找到子节点，回退到策略网络\n')
        genMovePolicy(go, willPlayColor)
        return
    
    # 检查是否有新的移动
    if len(bestNextNode.go.history) <= len(go.history):
        sys.stderr.write('MCTS搜索失败：没有新移动，回退到策略网络\n')
        genMovePolicy(go, willPlayColor)
        return
    
    bestMove = bestNextNode.go.history[-1]

    if debug:
        playoutResult = getPlayoutNetResult(go, willPlayColor)
        playoutMove = toPosition(torch.argmax(playoutResult))
        print(playoutMove, bestMove, playoutMove == bestMove)
        for child in root.children:
            print(child)

    # 输出搜索结果到stderr
    sys.stderr.write(f'MCTS搜索完成，候选移动:\n')
    for child in root.children:
        sys.stderr.write(str(child) + '\n')

    x, y = bestMove
    moveResult = go.move(willPlayColor, x, y)
    strPosition = toStrPosition(x, y)

    if moveResult == False:
        sys.stderr.write(f'Illegal move: {strPosition}\n')
        # 如果移动非法，也回退到策略网络
        go.board = root.go.board.copy()  # 恢复棋盘状态
        genMovePolicy(go, willPlayColor)
        return
    else:
        print(strPosition)
    return x, y


if __name__ == '__main__':
    # 测试代码
    go = Go()
    go.move(1, 3, 16)
    go.move(-1, 3, 3)
    go.move(1, 16, 16)
    go.move(-1, 16, 3)
    go.move(1, 2, 5)

    debug = True
    genMoveMCTS(go, -1, debug)

    for item in go.history:
        print(toStrPosition(item[0], item[1])) 