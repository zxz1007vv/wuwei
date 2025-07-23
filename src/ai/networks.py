import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.game import *

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

# 策略网络 - ResNet structure with increased parameters
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv_in = nn.Conv2d(15, 128, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(128)
        
        # 12 ResNet blocks with 128 channels
        self.res_blocks = nn.ModuleList([ResBlock(128) for _ in range(12)])
        
        # Policy head
        self.conv_policy = nn.Conv2d(128, 32, 1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.conv_final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        blank = x[:, 0]
        x = x.float()
        
        # Initial conv
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # ResNet blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Policy head
        x = F.relu(self.bn_policy(self.conv_policy(x)))
        x = self.conv_final(x)
        x = x.view(-1, 19 * 19)
        x = torch.cat((x * blank.view(-1, 19 * 19), torch.ones((len(x), 1)).to(x.device) * 1e-50), dim=1)
        return x

# 快速策略网络 - ResNet structure but lighter than PolicyNetwork
class PlayoutNetwork(nn.Module):
    def __init__(self):
        super(PlayoutNetwork, self).__init__()
        self.conv_in = nn.Conv2d(15, 64, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)

        # 4 ResNet blocks with 64 channels (lighter than PolicyNetwork)
        self.res_blocks = nn.ModuleList([ResBlock(64) for _ in range(4)])
        
        # Policy head (similar to original)
        self.conv_final = nn.Conv2d(64, 1, 1)
        self.linear = nn.Linear(19 * 19, 19 * 19 + 1)

    def forward(self, x):
        blank = x[:, 0]
        x = x.float()
        
        # Initial conv
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # ResNet blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Final convolution and linear layer
        x = self.conv_final(x)
        x = x.view(-1, 19 * 19)
        x = self.linear(x)
        x = torch.cat((x[:, :-1] * blank.view(-1, 19 * 19), x[:, -1:]), dim=1)
        x = F.log_softmax(x, dim=1)
        return x

# 价值网络 - ResNet structure
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv_in = nn.Conv2d(15, 64, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        
        # 12 ResNet blocks
        self.res_blocks = nn.ModuleList([ResBlock(64) for _ in range(12)])
        
        # Value head
        self.conv_value = nn.Conv2d(64, 32, 1)
        self.bn_value = nn.BatchNorm2d(32)
        self.conv_final = nn.Conv2d(32, 2, 1)
        self.linear = nn.Linear(2 * 19 * 19, 256)
        self.linear_final = nn.Linear(256, 1)

    def forward(self, x):
        x = x.float()
        
        # Initial conv
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # ResNet blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Value head
        x = F.relu(self.bn_value(self.conv_value(x)))
        x = self.conv_final(x)
        x = x.view(-1, 2 * 19 * 19)
        x = F.relu(self.linear(x))
        x = self.linear_final(x)
        x = x.view(-1)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    # Example usage
    policy_net = PolicyNetwork()
    playout_net = PlayoutNetwork()
    value_net = ValueNetwork()

    # Print model summaries
    print(policy_net)
    print(playout_net)
    print(value_net)

    # print model parameter counts in a human-readable format, eg. 1K, 1M, etc.
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Policy Network Parameters: {count_parameters(policy_net) / 1e6:.2f}M")
    print(f"Playout Network Parameters: {count_parameters(playout_net) / 1e6:.2f}M")
    print(f"Value Network Parameters: {count_parameters(value_net) / 1e6:.2f}M")
    
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(8, 15, 19, 19)  # Batch size of 8
    
    # Forward pass through the networks
    policy_output = policy_net(dummy_input)
    playout_output = playout_net(dummy_input)
    value_output = value_net(dummy_input)
    
    print("Policy Output Shape:", policy_output.shape)
    print("Playout Output Shape:", playout_output.shape)
    print("Value Output Shape:", value_output.shape)
