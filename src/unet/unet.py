import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embedding import TimeEmbedding
from .blocks import Down, Middle, Up

class UNet(nn.Module):

    def __init__(self, input_channels, channel_arr,
                block_arr, attn_arr, embed_dim,
                timesteps, dropout, padding=[0, 0, 0, 0]):

        """
        input_channels: int, the channels of the original input, e.g. 1 for MNIST, 3 for CIFAR.

        channel_arr: array of ints, resolution level channel outputs.

        block_arr: array of ints, number of residual blocks per resolution level.

        attn_arr: array of the form [[use_attn_1, num_heads_1], ..., [use_attn_n, num_heads_n]],
        where use_attn_i is True if the ith resolution level should use attention,
        and num_heads_i is the number of heads in the corresponding MultiHeadAttention layer
        (if use_attn_i == False, then num_heads_i is irrelevant).

        embed_dim: int, embedding dimension for the time embedding.

        timesteps: int, length of horizon for the diffusion model.

        dropout: float in range [0, 1], dropout rate for residual layers

        padding: array of ints, length of 4 describing how much the left, right,
        top and bottom of the image should be padded respectively.
        """
        
        super().__init__()

        assert len(channel_arr) == len(block_arr) == len(attn_arr)

        self.timesteps = timesteps
        self.padding = padding
        
        self.embedder = TimeEmbedding(embed_dim, timesteps)

        self.down = Down(input_channels, channel_arr, block_arr,
                        attn_arr, embed_dim, dropout)

        self.middle = Middle(channel_arr, block_arr, attn_arr,
                            embed_dim, dropout)

        self.up = Up(channel_arr, block_arr, attn_arr,
                    embed_dim, dropout)

        self.output = nn.ConvTranspose2d(channel_arr[0], input_channels, kernel_size=(3, 3), padding=(1, 1))
    
    
    def forward(self, x, t):
        
        pad_left, pad_right, pad_top, pad_bottom = self.padding
        x = F.pad(x, self.padding, "constant", 0)
        
        time_embed = self.embedder(t)
        
        out, down_outputs = self.down(x, time_embed)
        out = self.middle(out, time_embed)
        out = self.up(out, time_embed, down_outputs)

        out = self.output(out)

        if pad_bottom > 0:
            out = out[:, :, pad_top:-pad_bottom]
        else:
            out = out[:, :, pad_top:]

        if pad_right > 0:
            out = out[:, :, :, pad_left:-pad_right]
        else:
            out = out[:, :, :, pad_left:]
        
        return out