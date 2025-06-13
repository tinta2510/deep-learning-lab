import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 3, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            # When Conv is followed by BatchNorm, SET Conv's bias = False
            nn.Conv2d(in_channels, out_channels, stride = stride, padding = kernel_size // 2, 
                kernel_size = kernel_size, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, stride = 1, padding = kernel_size // 2, 
                kernel_size = kernel_size, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU(True)
    
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
    
class CIFARResNet(nn.Module):
    def __init__(self, in_channels, num_classes, filters = [16, 32, 64], blocks_per_stage = 3):
        super(CIFARResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size = 3, stride = 1, 
                padding = 1, bias = False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(True)
        )
        self.blocks_per_stage = blocks_per_stage
        self.curr_n_filters = filters[0] # Current output's num of filters
        
        layers = [self._make_layers(filters[0], stride = 1)]
        for i in filters[1:]:
            layers.append(self._make_layers(i, stride = 2))
        self.resnet = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters[-1], num_classes)
        
    def _make_layers(self, num_filters, stride = 1):
        shortcut = None
        if num_filters != self.curr_n_filters or stride != 1:
            shortcut = nn.Sequential(
                nn.Conv2d(self.curr_n_filters, num_filters, kernel_size = 1, 
                    stride = stride, bias = False),
                nn.BatchNorm2d(num_filters)
            )
        layers = []
        layers.append(ResidualBlock(self.curr_n_filters, num_filters, stride = stride, shortcut = shortcut))
        self.curr_n_filters = num_filters
        for _ in range(1, self.blocks_per_stage):
            layers.append(ResidualBlock(num_filters, num_filters))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.resnet(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1) # Flatten all dims except the batch dimension
        out = self.fc(out)
        return out