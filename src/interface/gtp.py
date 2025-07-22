from src.ai.networks import *
from src.core.game import *
import sys
from src.ai.engine import genMovePolicy, genMoveMCTS, colorCharToIndex, charToIndex

def main():
    go = Go()

    # 检查是否启用MCTS模式
    use_mcts = False
    if len(sys.argv) >= 3 and sys.argv[2] == 'MCTS':
        use_mcts = True
        sys.stderr.write('启动wuwei (MCTS模式)...\n')
    else:
        sys.stderr.write('启动wuwei (策略网络模式)...\n')

    # stderr output 'GTP ready'
    sys.stderr.write('GTP ready\n')

    while True:
        # implement GTP (Go Text Protocol)
        line = input().strip()
        if line == 'quit':
            break
        print('= ', end='')
        if line == 'boardsize 19':
            print('boardsize 19')
        elif line.startswith('boardsize'):
            print('boardsize')
        elif line.startswith('komi'):
            print('komi')
        elif line == 'clear_board':
            go = Go()
            print('clear_board')
        elif line.startswith('play'):
            # play B F12
            color, position = line.split()[1:]
            if position == 'pass':
                print('play PASS')
            else:
                # position = F12
                y, x = position[0], position[1:]

                #    A B C D E F G H J K L M N O P Q R S T
                # 19
                # 18
                # 17

                x = 19 - int(x)
                y = charToIndex[y]

                color = colorCharToIndex[color]

                if go.move(color, x, y) == False:
                    print('Illegal move')
                else:
                    print('ok')
        elif line.startswith('genmove'):
            colorChar = line.split()[1]
            willPlayColor = colorCharToIndex[colorChar]
            if use_mcts:
                genMoveMCTS(go, willPlayColor)
            else:
                genMovePolicy(go, willPlayColor)

        elif line.startswith('showboard'):
            for i in range(19):
                for j in range(19):
                    if go.board[i][j] == 1:
                        print('X', end='')
                    elif go.board[i][j] == -1:
                        print('O', end='')
                    else:
                        print('.', end='')
                print()
        # name
        elif line.startswith('name'):
            if use_mcts:
                print('wuwei (MCTS)')
            else:
                print('wuwei (策略网络)')
        # version
        elif line.startswith('version'):
            print('0.1')
        # protocol_version
        elif line.startswith('protocol_version'):
            print('2')
        # list_commands
        elif line.startswith('list_commands'):
            print('name')
            print('version')
            print('protocol_version')
            print('list_commands')
            print('clear_board')
            print('boardsize')
            print('showboard')
            print('play')
            print('genmove')
            print('quit')
        else:
            print('Unknown command')

        print()


if __name__ == '__main__':
    main()
