import torch
import torch.nn as nn
import timm

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(768, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        return x * attn

class LocationCNN(nn.Module):
    def __init__(self):
        super(LocationCNN, self).__init__()
        # Use EVA-02 base model pretrained on ImageNet
        self.features = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, features_only=True)
        
        # Freeze the features
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Add spatial attention
        self.spatial_attention = SpatialAttention()
        
        # Separate branches for latitude and longitude
        self.shared_layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4)
        )
        
        # Latitude-specific branch
        self.lat_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # Constrain latitude between -90 and 90 degrees
            nn.Tanh(),  # outputs [-1, 1]
        )
        
        # Longitude-specific branch
        self.lon_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # Constrain longitude between -180 and 180 degrees
            nn.Tanh(),  # outputs [-1, 1]
        )
    
    def forward(self, x):
        # Get features from EVA-02
        features = self.features(x)[-1]  # [B, 768, H, W]
        
        # Apply spatial attention
        features = self.spatial_attention(features)
        
        # Global average pooling
        x = torch.mean(features, dim=[2, 3])  # [B, 768]
        
        # Shared feature processing
        x = self.shared_layers(x)  # [B, 256]
        
        # Split into latitude and longitude branches
        lat = self.lat_branch(x) * 90.0  # Scale to [-90, 90]
        lon = self.lon_branch(x) * 180.0  # Scale to [-180, 180]
        
        return torch.cat([lat, lon], dim=1)