import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FTUCNNMulticlass(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.base_model = resnet18(weights = weights)

        # Replace first conv layer to accept 1-channel input (grayscale DICOM)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for param in self.base_model.parameters():
            param.requires_grad = False
        
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        for param in self.base_model.fc.parameters():
            param.requires_grad = True

        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    
    def forward(self, x):
        return self.base_model(x)
    
  
