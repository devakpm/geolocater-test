import torch

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