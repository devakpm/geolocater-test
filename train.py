import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
from model import LocationCNN
import pandas as pd

# Implementation of haversine formula to calculate distance between two coordinates on a sphere as loss function
def haversine_loss(y_pred, y_true):
    lat1 = y_pred[:,0] * torch.pi/180.0
    lon1 = y_pred[:,1] * torch.pi/180.0
    lat2 = y_true[:,0] * torch.pi/180.0
    lon2 = y_true[:,1] * torch.pi/180.0
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    return 6371 * c

def train_step(model, optimizer, images, coordinates, scaler):
    optimizer.zero_grad()
    with autocast():
        predictions = model(images)
        loss = haversine_loss(predictions, coordinates)
    
    scaler.scale(loss).backward() 
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocationCNN().to(device)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create data loader
    train_loader = create_data_loaders(
        image_dir="path/to/images",
        coordinates_file="path/to/coordinates.csv",
        batch_size=16
    )
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, coordinates) in enumerate(train_loader):
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            loss = train_step(model, optimizer, images, coordinates, scaler)
            running_loss += loss
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss:.4f}')
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1} complete, Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main()