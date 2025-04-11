import torch
import torch.nn as nn
from torch.nn import functional as F
import math 

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    # mask: avoid particular tokens
    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch_size, seq_len, dim)

        input_shape = x.shape  
        batch_size, sequence_leangth, d_embed = input_shape 

        intermin_shape = (batch_size, sequence_leangth, self.n_heads, self.d_head)

        # (Batch_size, seq_len, dim) -> (Batch_size, seq_len, dim * 3) -> 3 tensors of shape (Batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (Batch_size, seq_len, n_heads, dim // n_heads) -> (Batch_size, H, seq_len, dim // H)
        q = q.view(intermin_shape).transpose(1, 2)
        k = k.view(intermin_shape).transpose(1, 2)
        v = v.view(intermin_shape).transpose(1, 2)

        # (Batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # mask where the upper triangle
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, float('-inf'))
        
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (Batch_size, H, seq_len, dim / H)
        output = weight @ v

        # (Batch_size, seq_len, H, dim / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)
        output = self.out_proj(output)

        # (Batch_size, seq_len, dim)
        return output


