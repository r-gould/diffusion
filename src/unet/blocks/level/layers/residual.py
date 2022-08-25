import torch
import torch.nn as nn

class Residual(nn.Module):

    def __init__(self, c_in, c_out, embed_dim, dropout):

        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.GroupNorm(num_groups=32, num_channels=c_out),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.GroupNorm(num_groups=32, num_channels=c_out),
        )

        self.drop = nn.Dropout(dropout)

        if c_in != c_out:
            self.project_input = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        else:
            self.project_input = nn.Identity()

        self.project_time = nn.Linear(embed_dim, c_out)

    def forward(self, x, time_embed):

        out = self.block1(x)
        out += self.project_time(time_embed).unsqueeze(2).unsqueeze(2)
        out = self.drop(out)
        out = self.block2(out)
        return out + self.project_input(x)
        
