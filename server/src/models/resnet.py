import torch
from torchvision import models
from torch import nn

class ResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(ResNet, self).__init__()
        
        # Load ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.net = models.resnet18(weights=weights)
        
        # Modify the first conv layer to accept 1 channel (grayscale)
        original_first_layer = self.net.conv1
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.net.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
        
        # Freeze all layers initially
        for param in self.net.parameters():
            param.requires_grad = False
            
        # Unfreeze the last layer (layer4) and fc for fine-tuning
        for param in self.net.layer4.parameters():
            param.requires_grad = True
        for param in self.net.fc.parameters():
            param.requires_grad = True

        # Modify the fully connected layer
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.net(x)
