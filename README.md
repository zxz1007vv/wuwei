## wuwei


围棋 AI，仿照 Alphago，主要实现了简单的**策略网络**和**蒙特卡洛树搜索算法**

> Alphago 主要使用了卷积网络（策略网络、价值网络）和蒙特卡洛树搜索算法、强化学习等技术。

+ 策略网络输入是 features，输出是各个位置或 pass 的概率（对数）。使用卷积网络
+ 价值网络输入 features，输出胜率。也用到了卷积网络
+ 蒙特卡洛树搜索算法（MCTS）从策略网络输出的排名靠前的选点中选择合适的

> 其中 features 包括棋的位置、各个位置气的数量、最近的移动记录等信息。

AlphaGo 使用的是更多层卷积（包含残差网络）、更多 features，使用强化学习产生更多的数据。

由于计算资源的限制，Irene 只使用有监督的形式训练大约 2000 张棋谱。因为价值网络的预测较为不准确，MCTS 中使用棋子数作为评估指标。

### 项目结构

```
wuwei/
├── src/                    # 主要源代码
│   ├── core/              # 核心围棋逻辑
│   │   ├── game.py        # 围棋游戏规则
│   │   └── features.py    # 特征提取
│   ├── ai/                # AI相关模块
│   │   ├── networks.py    # 神经网络定义
│   │   ├── mcts.py        # MCTS算法
│   │   └── engine.py      # AI引擎
│   ├── data/              # 数据处理
│   │   ├── prepare.py     # 数据准备
│   │   └── filter.py      # SGF文件过滤
│   ├── training/          # 训练相关
│   │   └── trainer.py     # 训练脚本
│   └── interface/         # 接口层
│       └── gtp.py         # GTP协议
├── models/                # 训练好的模型
├── games/                 # 游戏数据（SGF文件）
├── tests/                 # 测试文件
├── scripts/               # 脚本文件
├── main.py                # 主程序入口
└── requirements.txt       # 依赖文件
```

### 使用方式

克隆项目并安装依赖

```bash
git clone https://github.com/GWDx/Irene.git
cd Irene
pip install -r requirements.txt
```

获取并处理数据

```bash
wget https://homepages.cwi.nl/~aeb/go/games/games.7z
7z x games.7z
python main.py filter-sgf
python main.py prepare-data
```

训练网络

```bash
python main.py train policy    # 训练策略网络
python main.py train playout   # 训练快速策略网络
python main.py train value     # 训练价值网络
```

运行AI

```bash
# 使用GTP协议运行（可以连接Sabaki等图形界面）
python main.py gtp

# 或直接运行GTP模块
python -m src.interface.gtp
```

运行测试

```bash
python tests/test_game.py
```

### 结果

+ 策略网络训练的正确率达到 38%，和 [Mugo](https://github.com/brilee/MuGo) 使用相同数据的正确率相当。但和 AlphaGo（57%）存在较大差距。
+ 对于价值网络，直接训练过拟合严重，AlphaGo 的论文中也提到了这个问题。这是他们使用了强化学习的一个原因。

策略网络（AI 执白）与人对弈的局面：

<img src="image/1.png" alt="1" width=60% />

+ 左上和左下都是合乎定式的下法（角上双方互不吃亏的下法），这是从棋谱中学到的
+ 但计算力较弱，没有计算出右上角的征子

MCTS（执白）与策略网络（执黑）对弈的结果，白棋胜：

<img src="image/2.png" alt="2" width=60% />

使用策略网络，并结合以棋子数为评估指标的 MCTS 搜索时，AI 倾向于吃子。相较于仅使用策略网络，计算力有所提高。但围棋更重要的是围空，吃子有时未必有利于围空。

### TODO

- [ ] 改进策略网络，考虑使用残差网络
- [ ] 训练策略网络时，增加训练使用的棋谱数量，可以尝试数据增强
- [ ] 使用强化学习训练价值网络

### 参考

+ [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
+ [Mugo GitHub](https://github.com/brilee/MuGo)
+ [ml_implementation MCTS GitHub](https://github.com/tobegit3hub/ml_implementation/blob/master/monte_carlo_tree_search/mcst_example.py)

### 感谢

+ [Sabaki](https://github.com/SabakiHQ/Sabaki)
+ [sgfmill](https://github.com/mattheww/sgfmill)
+ [28 天自制你的 AlphaGo 知乎](https://zhuanlan.zhihu.com/p/24885190)
