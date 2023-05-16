import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if (stride != 1) or (out_channels != in_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )



    def forward(self, x):
        out = self.features(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        num_classes = num_classes
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, stride=1)
        self.layer2 = self.make_block(out_channels=128, stride=2)
        self.layer3 = self.make_block(out_channels=256, stride=2)
        self.layer4 = self.make_block(out_channels=512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, stride):
        layers = []
        for stride in [stride, 1]:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class ResNet152(nn.Module):
    def __init__(self, num_classes=200):
        num_classes = num_classes
        super(ResNet152, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, num_blocks=3, stride=1)
        self.layer2 = self.make_block(out_channels=128, num_blocks=8, stride=2)
        self.layer3 = self.make_block(out_channels=256, num_blocks=16, stride=2)
        self.layer4 = self.make_block(out_channels=512, num_blocks=3, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class ResNet153(nn.Module):
    def __init__(self, num_classes=200):
        num_classes = num_classes
        super(ResNet152, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, num_blocks=3, stride=1)
        self.layer2 = self.make_block(out_channels=128, num_blocks=8, stride=2)
        self.layer3 = self.make_block(out_channels=256, num_blocks=16, stride=2)
        self.layer4 = self.make_block(out_channels=512, num_blocks=3, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class ResNetNEW(nn.Module):
    def __init__(self, num_classes=200):
        num_classes = num_classes
        super(ResNet152, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_block(out_channels=64, num_blocks=3, stride=1)
        self.layer2 = self.make_block(out_channels=128, num_blocks=8, stride=2)
        self.layer3 = self.make_block(out_channels=256, num_blocks=16, stride=2)
        self.layer4 = self.make_block(out_channels=512, num_blocks=3, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def make_block(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out, self.expansion * planes, kernel_size=1),
            nn.batchnorm(self.expansion * planes))
            
        self.shortcut = nn.Sequential()
        if (stride != 1) or (out_channels != in_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
            
    def forward(self, x):
        out = self.conv(x)
        out += shortcut(x)
        out = F.relu(OUT)
        return out