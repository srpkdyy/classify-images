import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1,
        groups=groups, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=4):
        super(ConvBlock, self).__init__()
        c = in_planes // groups
        self.conv1 = conv1x1(in_planes, c)
        self.conv2_1 = conv3x3(c, c)
        self.conv2_2 = conv3x3(c, c)
        self.conv2_3 = conv3x3(c, c)
        self.conv2_4 = conv3x3(c, c)
        self.conv3 = conv1x1(c, out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)


    def forward(self, x):
        x = self.conv1(x)

        #x = self.conv2_1(x)
        x1 = self.conv3(self.conv2_1(x))
        x2 = self.conv3(self.conv2_2(x))
        x3 = self.conv3(self.conv2_3(x))
        x4 = self.conv3(self.conv2_4(x))
        #x = self.conv3(x)

        x = x1 + x2 + x3 + x4
        x = self.bn(x)
        x = self.relu(x)
        return x



class CNN(nn.Module):
    def __init__(self, in_channels=3,  n_classes=10):
        super(CNN, self).__init__()
        #self.conv1 = ConvBlock(in_channels, 64)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 256)
        self.conv5 = ConvBlock(256, 512)
        self.conv6 = ConvBlock(512, 512)
        self.conv7 = ConvBlock(512, 512)
        self.conv8 = ConvBlock(512, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classify = nn.Linear(512, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classify(x)
        return x

