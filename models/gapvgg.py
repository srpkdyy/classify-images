import torch
import torch.nn as nn

from models import *

class GAPVGG(VGG):
    def __init__(self, n_classes=10):
        super(GAPVGG, self).__init__(vgg.make_layers(vgg.cfgs['16']), n_classes=n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
