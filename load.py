from torch.utils.data import DataLoader
from torchvision import transforms

def load(torch_dataset, input_channels, batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,) * input_channels, 
                             std=(0.5,) * input_channels),
    ])

    train_ds = torch_dataset(root="diffusion/data/", train=True, download=True, transform=transform)
    valid_ds = torch_dataset(root="diffusion/data/", train=False, download=True, transform=transform)
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=True)

    return {
        "train" : train_dl,
        "valid" : valid_dl
    }