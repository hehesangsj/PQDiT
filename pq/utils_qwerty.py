import torch
import torch.nn as nn

class SideNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SideNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out


class CompensationBlock(nn.Module):
    def __init__(self, block, dim, hidden_dim):
        super(CompensationBlock, self).__init__()
        self.block = block
        self.side_network = SideNetwork(dim, hidden_dim, dim)

    def forward(self, x, c):
        out = self.block(x, c)
        side_out = self.side_network(x)
        out = out + side_out

        return out
