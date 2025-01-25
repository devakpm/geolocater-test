import torch
from torch.amp import autocast, GradScaler
from model import LocationCNN
from utils import haversine_loss # Calculates distance as loss function

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
    scaler = GradScaler('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 100
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx in range(len(train_loader)):
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