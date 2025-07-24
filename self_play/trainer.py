import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.ai.networks import PolicyNetwork, ValueNetwork, PlayoutNetwork
from self_play.self_play_env import SelfPlayEnv

def setup_networks(models_dir, device='cuda'):
    """Initialize or load networks from checkpoints"""
    policy_net = PolicyNetwork().to(device)
    value_net = ValueNetwork().to(device)
    playout_net = PlayoutNetwork().to(device)
    
    # Load existing models if available
    if os.path.exists(os.path.join(models_dir, 'policyNet.pt')):
        policy_net.load_state_dict(torch.load(os.path.join(models_dir, 'policyNet.pt')))
    if os.path.exists(os.path.join(models_dir, 'valueNet.pt')):
        value_net.load_state_dict(torch.load(os.path.join(models_dir, 'valueNet.pt')))
    if os.path.exists(os.path.join(models_dir, 'playoutNet.pt')):
        playout_net.load_state_dict(torch.load(os.path.join(models_dir, 'playoutNet.pt')))
        
    return policy_net, value_net, playout_net

def train_policy(policy_net, states, policies, optimizer):
    """Train policy network"""
    policy_net.train()
    optimizer.zero_grad()
    
    # Forward pass
    policy_logits = policy_net(states)
    
    # Calculate loss (cross entropy between predicted policy and MCTS policy)
    loss = nn.CrossEntropyLoss()(policy_logits, policies)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_value(value_net, states, values, optimizer):
    """Train value network"""
    value_net.train()
    optimizer.zero_grad()
    
    # Forward pass
    value_preds = value_net(states)
    
    # Calculate loss (MSE between predicted value and actual game outcome)
    loss = nn.MSELoss()(value_preds, values)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_playout(playout_net, states, policies, optimizer):
    """Train playout network"""
    playout_net.train()
    optimizer.zero_grad()
    
    # Forward pass
    playout_logits = playout_net(states)
    
    # Calculate loss (cross entropy between predicted policy and MCTS policy)
    loss = nn.CrossEntropyLoss()(playout_logits, policies)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def save_checkpoint(policy_net, value_net, playout_net, checkpoint_dir, epoch):
    """Save network checkpoints"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save networks
    torch.save(policy_net.state_dict(), 
              os.path.join(checkpoint_dir, f'policy_net_epoch_{epoch}.pt'))
    torch.save(value_net.state_dict(), 
              os.path.join(checkpoint_dir, f'value_net_epoch_{epoch}.pt'))
    torch.save(playout_net.state_dict(), 
              os.path.join(checkpoint_dir, f'playout_net_epoch_{epoch}.pt'))
    
    # Also save as latest models
    torch.save(policy_net.state_dict(), os.path.join('models', 'policyNet.pt'))
    torch.save(value_net.state_dict(), os.path.join('models', 'valueNet.pt'))
    torch.save(playout_net.state_dict(), os.path.join('models', 'playoutNet.pt'))

def self_play_training(
    models_dir='models',
    checkpoint_dir='checkpoints',
    num_games=1000,
    num_epochs=10,
    games_per_epoch=100,
    batch_size=32,
    checkpoint_interval=10,
    save_games=True,
    policy_only=False,
    device='cuda'
):
    """Main self-play training loop"""
    # Setup networks and optimizers
    policy_net, value_net, playout_net = setup_networks(device)
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    playout_optimizer = optim.Adam(playout_net.parameters(), lr=0.001)
    
    # Setup learning rate schedulers
    policy_scheduler = optim.lr_scheduler.StepLR(policy_optimizer, step_size=5, gamma=0.1)
    value_scheduler = optim.lr_scheduler.StepLR(value_optimizer, step_size=5, gamma=0.1)
    playout_scheduler = optim.lr_scheduler.StepLR(playout_optimizer, step_size=5, gamma=0.1)
    
    # Create self-play environment
    env = SelfPlayEnv(policy_net, value_net, playout_net, device)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    if save_games:
        os.makedirs('game_records', exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Self-play phase
        for game in range(games_per_epoch):
            print(f"Playing game {game + 1}/{games_per_epoch}")
            
            state = env.reset()
            game_over = False
            
            while not game_over:
                # Get action probabilities (MCTS or policy network)
                policy, move_info = env.get_action_probs(policy_only)
                
                if policy_only:
                    if move_info == (None, None):
                        break  # Pass or no valid moves
                    last_move = move_info
                else:
                    # MCTS mode
                    if move_info is None:
                        break  # End game if no valid moves
                        
                    # Get the last move by comparing history lengths
                    if len(move_info.go.history) <= len(env.go.history):
                        break  # No new move was made
                        
                    last_move = move_info.go.history[-1]  # Get latest move
                
                # Execute move
                next_state, reward, game_over = env.step(last_move)
                
                # Store experience in replay buffer
                env.replay_buffer.push(state, policy, 
                                    torch.tensor(reward, device=device))
                
                state = next_state
            
            # Save game record if enabled
            if save_games:
                env.save_game_history(f'game_records/game_{epoch}_{game}.json')
        
        # Training phase
        if len(env.replay_buffer) >= batch_size:
            for _ in range(num_games // batch_size):
                # Sample from replay buffer
                states, policies, values = env.replay_buffer.sample(batch_size)

                # to device and ensure float32 dtype
                states = states.to(device).float()
                policies = policies.to(device)
                values = values.to(device).float()
                
                # Train networks
                policy_loss = train_policy(policy_net, states, policies, policy_optimizer)
                value_loss = train_value(value_net, states, values, value_optimizer)
                playout_loss = train_playout(playout_net, states, policies, playout_optimizer)
                
                print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                      f"Playout Loss: {playout_loss:.4f}")
        
        # Step schedulers
        policy_scheduler.step()
        value_scheduler.step()
        playout_scheduler.step()
        
        # Save checkpoints
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(policy_net, value_net, playout_net, checkpoint_dir, epoch + 1)

def main():
    """Main function"""
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Start training
    self_play_training(
        num_games=1000,
        num_epochs=10,
        games_per_epoch=100,
        batch_size=32,
        checkpoint_interval=2,
        save_games=True,
        policy_only=False
    )

if __name__ == '__main__':
    main()
