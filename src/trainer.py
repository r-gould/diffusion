import torch
import numpy as np

from tqdm import tqdm
from utils import save_samples

class Trainer:

    def __init__(self, dataloaders, optim, scheduler=None):
        
        self.dataloaders = dataloaders
        self.optim = optim
        self.scheduler = scheduler

    def train(self, model, epochs, validate=True, sample_count=25, 
              save_model=True, device="cuda"):

        train_dl = self.dataloaders.get("train")
        train_losses = []
        valid_losses = []

        for epoch in range(1, epochs+1):
            print("Epoch:", epoch)
            epoch_losses = []
            for images, _ in tqdm(train_dl):
                
                images = images.to(device)
                batch_size, _, _, _ = images.shape
                t = torch.randint(1, model.timesteps+1, (batch_size,)).to(device)

                pred_noise, actual_noise = model(images, t)
                loss = (actual_noise - pred_noise)**2
                loss = loss.mean()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_losses.append(loss.item())

            if save_model:
                print("Saving model...")
                torch.save(model.state_dict(), f"diffusion/saved/diffusion_{epoch}.pt")
                print("Model saved")
            
            if sample_count > 0:
                print("Generating samples...")
                shape = images.shape[1:]
                save_samples(model, sample_count, shape, idx=epoch)
                print("Samples generated")

            avg_loss = np.mean(epoch_losses[-50:])
            print("Avg. train loss:", avg_loss)
            train_losses.append(avg_loss)
            epoch_losses = []

            if validate:
                valid_dl = self.dataloaders.get("valid")
                valid_loss = self.test(model, valid_dl, device=device)
                print("Avg. valid loss:", valid_loss)
                valid_losses.append(valid_loss)
        
        return train_losses, valid_losses

    @torch.no_grad()
    def test(self, model, test_dl, device="cuda"):

        test_losses = []

        for images, _ in test_dl:

            images = images.to(device)
            batch_size, _, _, _ = images.shape
            t = torch.randint(1, model.timesteps+1, (batch_size,)).to(device)

            pred_noise, actual_noise = model(images, t)
            loss = (actual_noise - pred_noise)**2
            loss = loss.mean()

            test_losses.append(loss.item())

        return np.mean(test_losses)