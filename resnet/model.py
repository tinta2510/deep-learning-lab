import torch 
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, padding=1, 
                kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, stride=1, padding=1, 
                kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.conv2(out)
        if self.shortcut:
            residual = self.shortcut(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out