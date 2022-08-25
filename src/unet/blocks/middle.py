import torch
import torch.nn as nn

from .level import ResLevel

class Middle(nn.Module):

    def __init__(self, channel_arr, block_arr, attn_arr, 
                embed_dim, dropout):
        
        super().__init__()

        self.block = self.build_block(channel_arr, block_arr, attn_arr,
                                    embed_dim, dropout)

    def forward(self, x, time_embed):

        return self.block(x, time_embed)
        
    @staticmethod
    def build_block(channel_arr, block_arr,
                    attn_arr, embed_dim, dropout):

        c_in = channel_arr[-2]
        c_out = channel_arr[-1]
        num_blocks = block_arr[-1]
        use_attn, num_heads = attn_arr[-1]

        return ResLevel(c_in, c_out, num_blocks, use_attn, 
                        num_heads, embed_dim, dropout)