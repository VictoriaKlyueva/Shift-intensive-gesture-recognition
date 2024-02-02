import torch.nn as nn
from torchvision import models


class ResNet34(nn.Module):
    def __init__(self, num_classes=25):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.drop(x)
        x = self.fc(x)
        return x
