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


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent): (batch_size, seq_len_q, dim_q)
        # y: (context): (batch_sizem seq_len_kv, dim_kv) = (batch_size, 77, 768)

        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x) 
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)

        output = self.out_proj(output)
        return output