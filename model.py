import torch.nn as nn
from torchvision import models

class ResNet50MultiLabel(nn.Module):
    def __init__(self, num_classes=8, pretrained=False):
        super(ResNet50MultiLabel, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)