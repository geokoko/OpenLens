import torch
from torchvision import models
from torch import nn

class MobileNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(MobileNet, self).__init__()
        
        # Load MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.net = models.mobilenet_v2(weights=weights)
        
        # Modify the first layer to accept 1 channel (grayscale) instead of 3
        # We average the weights of the original 3 channels to initialize the single channel
        original_first_layer = self.net.features[0][0]
        self.net.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.net.features[0][0].weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
        
        # Freeze early layers
        for param in self.net.features.parameters():
            param.requires_grad = False
            
        # Unfreeze the last few blocks for fine-tuning
        for param in self.net.features[-3:].parameters():
            param.requires_grad = True

        # Modification of the classifier
        self.net.classifier[1] = nn.Linear(self.net.last_channel, num_classes)
    

    def forward(self, x):
        return self.net(x)
