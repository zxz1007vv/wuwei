import torch
import numpy as np
import sys
import os
from src.ai.networks import PolicyNetwork, PlayoutNetwork, ValueNetwork
from src.core.game import Go, toPosition, toStrPosition
from src.core.features import getAllFeatures
from src.ai.mcts import MCTSNode, MCTS

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Engine:

    def __init__(self, path=None):
        # Set random seeds
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)

        # Get program path
        path = path if path else os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))) + '/models'

        # Load pre-trained models
        self.policy_net = PolicyNetwork()
        self.policy_net.load_state_dict(
            torch.load(os.path.join(path, 'policyNet.pt')))
        self.policy_net.to(device)
        self.policy_net.eval()

        self.playout_net = PlayoutNetwork()
        self.playout_net.load_state_dict(
            torch.load(os.path.join(path, 'playoutNet.pt')))
        self.playout_net.to(device)
        self.playout_net.eval()

        self.value_net = ValueNetwork()
        self.value_net.load_state_dict(torch.load(os.path.join(path, 'valueNet.pt')))
        self.value_net.to(device)
        self.value_net.eval()

    @torch.no_grad()
    def get_policy_net_result(self, go, will_play_color):
        """Get policy network prediction"""
        input_data = getAllFeatures(go, will_play_color)
        input_data = torch.tensor(input_data).bool().reshape(1, -1, 19, 19).to(device)
        predict = self.policy_net(input_data)[0].detach().cpu()
        return predict

    @torch.no_grad()
    def get_playout_net_result(self, go, will_play_color):
        """Get playout network prediction"""
        input_data = getAllFeatures(go, will_play_color)
        input_data = torch.tensor(input_data).bool().reshape(1, -1, 19, 19).to(device)
        predict = self.playout_net(input_data)[0].detach().cpu()
        return predict

    @torch.no_grad()
    def get_value_net_result(self, go, will_play_color):
        """Get value network prediction"""
        input_data = getAllFeatures(go, will_play_color)
        input_data = torch.tensor(input_data).bool().reshape(1, -1, 19, 19).to(device)
        value = self.value_net(input_data)[0].detach().cpu().item()
        return value

    def get_value_result(self, go, will_play_color):
        """Get simple value evaluation (based on piece count difference)"""
        count_this_color = np.sum(go.board == will_play_color)
        count_another_color = np.sum(go.board == -will_play_color)
        return count_this_color - count_another_color

    def gen_move_policy(self, go, will_play_color):
        """Generate move using policy network"""
        predict = self.get_policy_net_result(go, will_play_color)
        predict_reverse_sort_index = reversed(torch.argsort(predict))

        # Output valueNet result to stderr
        value = self.get_value_result(go, will_play_color)
        sys.stderr.write(f'{will_play_color} {value}\n')

        # Output Candidate moves to stderr
        sys.stderr.write('Policy Candidate moves:\n')
        for predict_index in predict_reverse_sort_index[:5]:
            x, y = toPosition(predict_index)
            if (x, y) == (None, None):
                sys.stderr.write('pass\n')
            else:
                str_position = toStrPosition(x, y)
                sys.stderr.write(f'{str_position} {predict[predict_index].item()}\n')

        # TODO: manually select a move
        for predict_index in predict_reverse_sort_index:
            x, y = toPosition(predict_index)
            if (x, y) == (None, None):
                sys.stderr.write('pass\n')
                continue
            move_result = go.move(will_play_color, x, y)
            str_position = toStrPosition(x, y)

            if move_result == False:
                sys.stderr.write(f'Illegal move ({x}, {y}): {str_position}\n')
                continue
            else:
                print(str_position)
                return x, y
        print('pass')
        return None, None

    def gen_move_mcts(self, go, will_play_color, debug=False):
        """Generate move using MCTS"""
        root = MCTSNode(go, will_play_color, None)

        # TODO: manually select a move
        best_next_node = MCTS(
            root,
            self.get_policy_net_result,
            self.get_playout_net_result,
            self.get_value_net_result,
            debug=debug
        )

        # Fallback to policy network if MCTS search fails
        if best_next_node is None:
            sys.stderr.write(
                'MCTS search failed: no child nodes found, falling back to policy network\n')
            return self.gen_move_policy(go, will_play_color)

        # Check if there's a new move
        if len(best_next_node.go.history) <= len(go.history):
            sys.stderr.write(
                'MCTS search failed: no new moves, falling back to policy network\n')
            return self.gen_move_policy(go, will_play_color)

        best_move = best_next_node.go.history[-1]

        if debug:
            playout_result = self.get_playout_net_result(go, will_play_color)
            playout_move = toPosition(torch.argmax(playout_result))
            print(playout_move, best_move, playout_move == best_move)

        # Output search results to stderr
        sys.stderr.write(f'MCTS search complete, candidate moves:\n')
        for child in root.children:
            sys.stderr.write(str(child) + '\n')

        x, y = best_move
        move_result = go.move(will_play_color, x, y)
        str_position = toStrPosition(x, y)

        if move_result == False:
            sys.stderr.write(f'Illegal move ({x}, {y}): {str_position}\n')
            # Fallback to policy network if move is illegal
            go.board = root.go.board.copy()  # Restore board state
            return self.gen_move_policy(go, will_play_color)
        else:
            print(str_position)
        return x, y


if __name__ == '__main__':
    # Test code
    engine = Engine()
    go = Go()
    go.move(1, 3, 16)
    go.move(-1, 3, 3)
    go.move(1, 16, 16)
    go.move(-1, 16, 3)
    go.move(1, 2, 5)

    debug = True
    engine.gen_move_mcts(go, -1, debug)

    for item in go.history:
        print(toStrPosition(item[0], item[1]))
