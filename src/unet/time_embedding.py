import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):

    def __init__(self, embed_dim, timesteps, scaling=4):
        
        super().__init__()

        self.embedding = self.init_embedding(embed_dim // scaling, timesteps)
        self.network = nn.Sequential(
            nn.Linear(embed_dim // scaling, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t):

        embeds = self.embedding[t].to(t.device)
        return self.network(embeds)

    @staticmethod
    def init_embedding(embed_dim, timesteps):

        encoding = torch.zeros(timesteps, embed_dim)
        pos = torch.arange(0, timesteps).reshape(-1, 1)
        idx = torch.arange(0, embed_dim)

        encoding[:, 0::2] = torch.sin(pos / 10000**(idx[0::2] / embed_dim))
        encoding[:, 1::2] = torch.cos(pos / 10000**((idx[1::2]-1) / embed_dim))
        return encoding