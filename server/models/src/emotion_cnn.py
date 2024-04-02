import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # Load a pre-trained ResNet-18 model. Inittially, the model is trained on ImageNet dataset
        self.resnet = models.resnet18(pretrained=True)
        
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the fully connected layers of ResNet-18 and use it purely as a feature extractor
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Custom layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)  # ResNet-18 features to 256
        self.fc2 = nn.Linear(256, num_classes)  # Final output layer



    def forward(self, x):
        # Feature extraction
        x = self.resnet(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # Custom classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25, training=self.training)
        x = self.fc2(x)
        return x
