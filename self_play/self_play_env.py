import torch
import numpy as np
from src.core.game import Go, toPosition, toDigit
from src.core.features import getAllFeatures
from src.ai.mcts import MCTSNode, MCTS

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.states = []  # Board states
        self.policies = []  # Policy targets (MCTS visit counts)
        self.values = []  # Value targets (final game results)
        self.position = 0
        
    def push(self, state, policy, value):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.policies.append(policy)
            self.values.append(value)
        else:
            self.states[self.position] = state
            self.policies[self.position] = policy 
            self.values[self.position] = value
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size)
        states = torch.stack([self.states[i] for i in indices])
        policies = torch.stack([self.policies[i] for i in indices])
        values = torch.stack([self.values[i] for i in indices])
        return states, policies, values
        
    def __len__(self):
        return len(self.states)

class SelfPlayEnv:
    def __init__(self, policy_net, value_net, playout_net, device='cuda'):
        self.device = device
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.playout_net = playout_net.to(device)
        self.go = Go()
        self.replay_buffer = ReplayBuffer()
        self.current_color = 1  # Black starts
        self.history = []
        
    def reset(self):
        self.go = Go()
        self.current_color = 1
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        return torch.tensor(getAllFeatures(self.go, self.current_color)).bool()
    
    def _get_valid_moves(self):
        valid_moves = []
        for x in range(19):
            for y in range(19):
                # Use temporary board to check move validity
                temp_go = self.go.clone()
                if temp_go.move(self.current_color, x, y):
                    valid_moves.append((x, y))
        return valid_moves
    
    @torch.no_grad()
    def get_policy(self, go, will_play_color):
        """Get policy network prediction"""
        input_data = getAllFeatures(go, will_play_color)
        input_data = torch.tensor(input_data).bool().reshape(1, -1, 19, 19).to(self.device)
        predict = self.policy_net(input_data)[0].detach().cpu()
        return predict
    @torch.no_grad()
    def get_playout_policy(self, go, will_play_color):
        """Get playout network prediction"""
        input_data = getAllFeatures(go, will_play_color)
        input_data = torch.tensor(input_data).bool().reshape(1, -1, 19, 19).to(self.device)
        predict = self.playout_net(input_data)[0].detach().cpu()
        return predict
    @torch.no_grad()
    def get_value(self, go, will_play_color):
        """Get value network prediction"""
        input_data = getAllFeatures(go, will_play_color)
        input_data = torch.tensor(input_data).bool().reshape(1, -1, 19, 19).to(self.device)
        value = self.value_net(input_data)[0].detach().cpu().item()
        return value
    
    @torch.no_grad()
    def get_policy_action(self):
        """Get move probabilities directly from policy network"""
        policy = self.get_policy(self.go, self.current_color)
        
        # Get valid moves
        valid_moves = self._get_valid_moves()
        
        # Find best valid move
        best_move = None
        best_prob = float('-inf')
        
        for x, y in valid_moves:
            prob = policy[toDigit(x, y)]
            if prob > best_prob:
                best_prob = prob
                best_move = (x, y)
        
        # If no valid moves found, pass
        if best_move is None:
            best_move = (None, None)
            policy = torch.zeros(19*19 + 1)
            policy[-1] = 1.0  # All probability on pass move
        
        return policy, best_move
    
    @torch.no_grad()
    def get_action_probs(self, policy_only=False):
        """Get move probabilities using MCTS or policy network"""
        if policy_only:
            return self.get_policy_action()
        
        # Use MCTS
        root = MCTSNode(self.go, self.current_color, None)
        
        # Run MCTS search with network prediction functions
        best_node = MCTS(
            root,
            self.get_policy,
            self.get_playout_policy,
            self.get_value,
            iterations=200  # Default MCTS iterations
        )
        
        # Extract policy from visit counts
        policy = torch.zeros(19*19 + 1)  # Include pass move
        total_visits = sum(c.N for c in root.children) if root.children else 1
        
        for child in root.children:
            # Get the last move from child's go history
            if len(child.go.history) > len(root.go.history):
                x, y = child.go.history[-1]
                policy[toDigit(x, y)] = child.N / total_visits
            else:  # Pass move
                policy[-1] = child.N / total_visits
                
        # Get best move from best node
        if best_node and len(best_node.go.history) > len(root.go.history):
            best_move = best_node.go.history[-1]
        else:
            best_move = (None, None)  # Pass move
                
        return policy, best_node
    
    def step(self, move):
        """Execute a move and return (next_state, reward, done)"""
        x, y = move
        
        # Store current state
        state = self._get_state()
        
        # Make move
        if not self.go.move(self.current_color, x, y):
            raise ValueError(f"Invalid move: ({x}, {y})")
            
        self.history.append((x, y, self.current_color))
        
        # Check if game is over (two consecutive passes)
        last_two_moves = self.go.history[-2:]
        game_over = all(move == (None, None) for move in last_two_moves)
        
        # Also check if no valid moves left
        if not game_over:
            valid_moves = self._get_valid_moves()
            game_over = len(valid_moves) == 0
        
        # Calculate immediate reward
        reward = self._calculate_reward()
        
        # Switch players
        self.current_color *= -1
        
        next_state = self._get_state()
        
        return next_state, reward, game_over
        
    def _calculate_reward(self):
        """Calculate reward for the current state"""
        # Territory-based reward
        black_territory = np.sum(self.go.board == 1)
        white_territory = np.sum(self.go.board == -1)
        
        # Favor center control
        center_x, center_y = 9, 9
        center_influence = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = center_x + dx, center_y + dy
                if 0 <= x < 19 and 0 <= y < 19:
                    center_influence += self.go.board[x][y] * self.current_color
                    
        # Liberty-based reward
        liberty_reward = 0
        for x in range(19):
            for y in range(19):
                if self.go.board[x][y] == self.current_color:
                    liberty_reward += self.go.liberty[x, y]
                    
        # Combine rewards
        reward = (
            0.5 * (black_territory - white_territory) * self.current_color +
            0.3 * center_influence +
            0.2 * liberty_reward
        )
        
        return reward
    
    def save_game_history(self, filename):
        """Save game history to file"""
        import json
        history_data = {
            'moves': [(x, y, color) for x, y, color in self.history],
            'final_board': self.go.board.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(history_data, f)
