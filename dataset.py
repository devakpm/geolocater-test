import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class GeoLocationDataset(Dataset):
    def __init__(self, image_dir, coordinates_file, transform=None):
        self.image_dir = image_dir
        # TODO: Load coordinates from file
        self.coordinates = []  # Will contain (lat, lon) pairs
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(528),  # Resize shorter edge to 528
                transforms.CenterCrop(528),  # Crop to square, we do not want to distort the image since the model is trained on square images
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        # TODO: Load image path and coordinates
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        coordinates = self.coordinates[idx]

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(coordinates)

# Function to create data loaders
def create_data_loaders(image_dir, coordinates_file, batch_size=16, num_workers=4):
    dataset = GeoLocationDataset(image_dir, coordinates_file)
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader