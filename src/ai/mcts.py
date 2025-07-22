import numpy as np
import torch
import sys
from src.core.game import toPosition


class MCTSNode:
    def __init__(self, go, willPlayColor, parent):
        self.go = go.clone()
        self.color = willPlayColor
        self.parent = parent
        self.children = []
        self.N = 0  # visit count
        self.Q = 0  # win rate
        self.expanded = False
        if parent:
            self.parent.children.append(self)

    def UCB(self):
        if self.N == 0:
            return float('inf')  # 未访问过的节点优先级最高
        if self.parent is None or self.parent.N == 0:
            return 0
        return self.Q / self.N + np.sqrt(2 * np.log(self.parent.N) / self.N)

    def __str__(self):
        from .engine import toStrPosition
        if len(self.go.history) == 0:
            strPosition = "root"
        else:
            x, y = self.go.history[-1]
            strPosition = toStrPosition(x, y)
        result = f'{self.color} {self.N} {self.Q:.3f} {self.UCB():.3f} {strPosition}'
        return result


def getBestChild(node):
    """选取UCB最大的节点"""
    bestChild = None
    bestUCB = float('-inf')
    for child in node.children:
        ucb = child.UCB()
        if ucb > bestUCB:
            bestChild = child
            bestUCB = ucb
    return bestChild


def getMostVisitedChild(node):
    """获取访问次数最多的子节点"""
    bestChild = None
    bestN = 0
    for child in node.children:
        if child.N > bestN:
            bestChild = child
            bestN = child.N
    return bestChild


def searchChildren(node, getPolicyNetResult):
    """为节点搜索子节点"""
    go = node.go
    nodeWillPlayColor = node.color

    predict = getPolicyNetResult(go, nodeWillPlayColor)
    predictReverseSortIndex = reversed(torch.argsort(predict))

    count = 0
    nextColor = -nodeWillPlayColor

    # 移除pass的检查，确保至少创建一些子节点
    for predictIndex in predictReverseSortIndex:
        x, y = toPosition(predictIndex)
        if (x, y) == (None, None):
            continue
        newGo = go.clone()

        if newGo.move(nodeWillPlayColor, x, y):
            newNode = MCTSNode(newGo, nextColor, node)
            count += 1
            if count >= 5:  # 增加候选子节点数量
                break


def treePolicy(root):
    """
    传入当前开始搜索的节点，返回创建的新的节点
    先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找UCB最大的节点
    """
    node = root
    while True:
        if len(node.children) == 0:
            return node

        # 寻找未访问的子节点
        unvisited = [child for child in node.children if child.N == 0]
        if unvisited:
            return unvisited[0]  # 返回第一个未访问的节点
        else:
            node = getBestChild(node)
            if node is None:  # 防止无限循环
                return root


def backward(node, value):
    """反向传播MCTS搜索结果"""
    while node:
        node.N += 1
        node.Q += value
        node.expanded = True
        node = node.parent


def defaultPolicy(expandNode, rootColor, getPlayoutNetResult, getValueNetResult, debug=False):
    """随机操作后创建新的节点，返回最终节点的value"""
    newGo = expandNode.go.clone()
    willPlayColor = expandNode.color

    for i in range(5):
        predict = getPlayoutNetResult(newGo, willPlayColor)

        attempts = 0
        while attempts < 20:  # 防止无限循环
            # random choose a move
            selectedIndex = np.random.choice(len(predict), p=predict.exp().detach().numpy())
            x, y = toPosition(selectedIndex)
            if (x, y) == (None, None):
                break  # pass move
            if newGo.move(willPlayColor, x, y):
                break
            attempts += 1

        willPlayColor = -willPlayColor

    value = getValueNetResult(newGo, rootColor)

    if debug:
        print(f'expandNode: {expandNode} value: {value}')

    return value


def MCTS(root, getPolicyNetResult, getPlayoutNetResult, getValueNetResult, iterations=200, debug=False):
    """执行MCTS搜索"""
    rootColor = root.color
    for i in range(iterations):
        expandNode = treePolicy(root)
        if expandNode is None:
            break
        searchChildren(expandNode, getPolicyNetResult)
        value = defaultPolicy(expandNode, rootColor, getPlayoutNetResult, getValueNetResult, debug)
        backward(expandNode, value)

    # 选择访问次数最多的子节点，而不是UCB最大的
    bestNextNode = getMostVisitedChild(root)
    return bestNextNode 