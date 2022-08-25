import yaml
import torch

from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import MNIST, CIFAR10

from src.diffusion import Diffusion
from src.unet import UNet
from src.trainer import Trainer
from utils import plot_stats
from load import load

DATASETS = {
    "mnist" : (MNIST, 1, (2, 2, 2, 2)), # 28x28 -> 32x32
    "cifar" : (CIFAR10, 3, (0, 0, 0, 0)),
}

def main(unet_params, batch_size, max_lr, warmup_steps, epochs, 
         data_str, load_model=True, save_model=True, device="cuda"):

    assert data_str in DATASETS.keys()

    torch_dataset, input_channels, padding = DATASETS[data_str]
    dataloaders = load(torch_dataset, input_channels, batch_size)

    network = UNet(**unet_params, input_channels=input_channels,
                   padding=padding)
    model = Diffusion(network, device).to(device)

    if load_model:
        model.load_state_dict(torch.load("saved/diffusion.pt"))

    optim = torch.optim.Adam(model.parameters(), lr=max_lr)
    warmup_func = lambda step: min(1, step / warmup_steps)
    scheduler = LambdaLR(optim, warmup_func)

    trainer = Trainer(dataloaders, optim, scheduler)
    train_losses, valid_losses = trainer.train(model, epochs, validate=True,
                                               save_model=save_model, device=device)

    plot_stats(train_losses, valid_losses)
    return model

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("unet.yaml", "r") as stream:
        unet_params = yaml.safe_load(stream)

    batch_size = 128
    max_lr = 2e-4
    warmup_steps = 5000
    epochs = 50
    
    model = main(unet_params, batch_size, max_lr, warmup_steps, 
                epochs, data_str="mnist", load_model=False, 
                save_model=True, device=device)
