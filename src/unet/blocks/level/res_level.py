import torch
import torch.nn as nn

from .layers.residual import Residual
from .layers.attention import Attention

class ResLevel(nn.Module):

    def __init__(self, c_in, c_out, num_blocks, use_attn, 
                num_heads, embed_dim, dropout):

        super().__init__()

        self.level = self.build_level(c_in, c_out, num_blocks, use_attn, 
                                    num_heads, embed_dim, dropout)
    
    def forward(self, x, time_embed):
        
        out = x
        for layer in self.level:
            out = layer(out, time_embed)

        return out

    @staticmethod
    def build_level(c_in, c_out, num_blocks, use_attn, 
                    num_heads, embed_dim, dropout):

        layers = [Residual(c_in, c_out, embed_dim, dropout)]
        
        for _ in range(num_blocks-1):
            if use_attn:
                layers.append(Attention(c_out, num_heads))
            layers.append(Residual(c_out, c_out, embed_dim, dropout))

        return nn.ModuleList(layers)