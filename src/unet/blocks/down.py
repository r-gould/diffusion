import torch
import torch.nn as nn

from .level import ResLevel

class Down(nn.Module):

    def __init__(self, input_channels, channel_arr, block_arr,
                attn_arr, embed_dim, dropout):

        super().__init__()

        self.block = self.build_block(input_channels, channel_arr, block_arr, 
                                    attn_arr, embed_dim, dropout)

    def forward(self, x, time_embed):
        
        down_outputs = []
        out = x
        for layer in self.block:
            if isinstance(layer, nn.MaxPool2d):
                down_outputs.append(out)
                out = layer(out)
                continue
            
            out = layer(out, time_embed)

        return out, down_outputs

    @staticmethod
    def build_block(input_channels, channel_arr, block_arr,
                    attn_arr, embed_dim, dropout):

        block = []
        num_levels = len(channel_arr)
        c_in = input_channels
        for i in range(num_levels-1):
            c_out = channel_arr[i]
            num_blocks = block_arr[i]
            use_attn, num_heads = attn_arr[i]

            block.extend([
                ResLevel(c_in, c_out, num_blocks, use_attn, 
                        num_heads, embed_dim, dropout),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ])
            
            c_in = c_out

        return nn.ModuleList(block)
