import math
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid

def save_samples(model, count, shape, idx=0):
    
    x_0 = model.sample(count, shape)
    nrow = math.ceil(math.sqrt(count))
    grid = make_grid(x_0, nrow=nrow)
    save_image(grid, f"diffusion/images/generated/grid_{idx}.png")

def plot_stats(train_losses, valid_losses):

    plt.title("Train losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(train_losses)
    plt.savefig("diffusion/images/plots/train.png")
    plt.show()

    plt.title("Valid losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(valid_losses)
    plt.savefig("diffusion/images/plots/valid.png")
    plt.show()

    plt.title("Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Valid")
    plt.legend(loc="upper right")
    plt.savefig("diffusion/images/plots/both.png")
    plt.show()