import torch
from torchvision import models, transforms, datasets
from torch import nn, optim

class MobileNet (nn.Module):
    def __init__(self, n_classes=7):
        super(MobileNet, self).__init__()
        
        self.net = models.mobilenet_v2(pretrained=True)
        self.num_classes = n_classes
        # Freeze all the layers in the pretrained model

        for param in self.net.parameters():
            param.requires_grad = False

        # Modification of the classsifier to adapt to the problem
        self.net.classifier[1] = nn.Linear(self.net.last_channel, self.num_classes)
    

    def forward(self, x):
        return self.net(x)