import torch
import torch.nn as nn

class Diffusion(nn.Module):

    def __init__(self, network, device="cuda"):

        super().__init__()
        
        self.network = network
        self.timesteps = network.timesteps

        self.betas = torch.linspace(start=1e-4, end=2e-2, steps=self.timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(self.betas)
        self.device = device

    def forward(self, x_0, t):
        
        t_idx = t - 1
        x_t, noise = self.add_noise(x_0, t)
        return self.network(x_t, t_idx), noise

    def add_noise(self, x_0, t):
        
        t_idx = t - 1
        alpha_bars_t = self.alpha_bars[t_idx].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bars_t) * x_0 + torch.sqrt(1 - alpha_bars_t) * noise
        return x_t, noise

    def step_backward(self, x_t, t, add_noise=True):
        
        t_idx = t - 1
        pred_noise = self.network(x_t, t_idx)
        coeff = self.betas[t_idx] / torch.sqrt(1 - self.alpha_bars[t_idx])
        coeff = coeff.reshape(-1, 1, 1, 1)

        z = 0
        if add_noise:
            z = torch.randn_like(x_t)

        alphas_t = self.alphas[t_idx].reshape(-1, 1, 1, 1)
        sigmas_t = self.sigmas[t_idx].reshape(-1, 1, 1, 1)
        
        mean = (x_t - coeff * pred_noise) / torch.sqrt(alphas_t)
        return mean + sigmas_t * z

    @torch.no_grad()
    def sample(self, count, image_shape):

        x_T = torch.randn(count, *image_shape).to(self.device)
        x_t = x_T
        for t in range(self.timesteps, 0, -1):
            add_noise = (t > 1)
            t_tensor = t * torch.ones(count, dtype=torch.int64).to(self.device)
            x_t = self.step_backward(x_t, t_tensor, add_noise)
        x_0 = x_t
        return x_0