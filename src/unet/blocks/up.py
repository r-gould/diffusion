import torch
import torch.nn as nn

from .level import ResLevel

class Up(nn.Module):

    def __init__(self, channel_arr, block_arr, attn_arr,
                embed_dim, dropout):

        super().__init__()

        self.block = self.build_block(channel_arr, block_arr, attn_arr,
                                    embed_dim, dropout)

    def forward(self, x, time_embed, down_outputs):
        
        out = x
        for layer in self.block:
            if isinstance(layer, nn.ConvTranspose2d):
                out = layer(out)
                down_out = down_outputs.pop()
                out = torch.cat((out, down_out), dim=1)
                continue

            out = layer(out, time_embed)

        return out

    @staticmethod
    def build_block(channel_arr, block_arr,
                    attn_arr, embed_dim, dropout):

        block = []
        num_levels = len(channel_arr)
        c_in = channel_arr[-1]
        for i in reversed(range(num_levels-1)):
            c_out = channel_arr[i]
            num_blocks = block_arr[i]
            use_attn, num_heads = attn_arr[i]

            block.extend([
                nn.ConvTranspose2d(c_in, c_out, kernel_size=(2, 2), stride=(2, 2)),
                ResLevel(2*c_out, c_out, num_blocks, use_attn, 
                        num_heads, embed_dim, dropout),
            ])
            
            c_in = c_out

        return nn.ModuleList(block)