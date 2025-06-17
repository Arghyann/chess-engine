import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNN(nn.Module):
    def __init__(self, input_planes=19, output_size=4672):
        super(ChessNN, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, hidden_size=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x