import torch
import torch.nn as nn
import torchvision.models as models

class LocationCNN(nn.Module):
    def __init__(self):
        super(LocationCNN, self).__init__()
        # largest model that can be trained locally, will maybe switch to a larger model and train on cloud
        efficientnet = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1) 
        self.features = efficientnet.features
        # Freeze the features
        for param in self.features.parameters():
            param.requires_grad = False
        # Replace the last layer with a new one
        self.regressor = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flattens 4D tensor to 2D
        x = self.regressor(x)
        return x