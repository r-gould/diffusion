import torch
import numpy as np
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, c_in, num_heads, d_v=None, d_k=None):
        
        super().__init__()

        if d_v is None:
            d_v = c_in
        if d_k is None:
            d_k = c_in

        self.attention = MultiHeadAttention(num_heads, c_in, d_v, d_k)

    def forward(self, x, time_embed=None):

        batch_size, c_in, h, w = x.shape
        x = x.view(batch_size, c_in, -1).permute(0, 2, 1)
        attn = self.attention(x, x, x)
        attn += x
        attn = attn.permute(0, 2, 1).view(batch_size, c_in, h, w)
        return attn

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, d_v, d_k):
        
        super().__init__()

        self.num_heads = num_heads

        self.linear_V = nn.Linear(d_model, num_heads * d_v)
        self.linear_K = nn.Linear(d_model, num_heads * d_k)
        self.linear_Q = nn.Linear(d_model, num_heads * d_k)

        self.attention = ScaledDotProductAttention(d_k)

        self.output = nn.Linear(num_heads * d_v, d_model)

    def forward(self, Q, K, V):
        
        inp = (self.linear_Q(Q), self.linear_K(K), self.linear_V(V))
        inp = map(self.split, inp)

        attn = self.attention(*inp)
        out = self.concat(attn)
        return self.output(out)

    def split(self, batch):
        # batch of shape (batch_size, seq_len, num_heads*d)
        
        batch_size, seq_len, _ = batch.shape
        batch = batch.reshape(batch_size, seq_len, self.num_heads, -1)
        return batch.permute(0, 2, 1, 3)

    def concat(self, batch):
        # batch of shape (batch_size, num_heads, seq_len, d_v)

        batch_size, _, seq_len, _ = batch.shape
        batch = batch.permute(0, 2, 1, 3)
        return batch.reshape(batch_size, seq_len, -1)

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k):
        
        super().__init__()

        self.d_k = d_k

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):

        scores = Q @ K.transpose(-1, -2) / np.sqrt(self.d_k)
        return self.softmax(scores) @ V