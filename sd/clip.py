import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens:int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))
    
    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding
        return x 

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention  = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, n_embd * 4)
        self.linear_2 = nn.Linear(n_embd * 4, n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (batch_size, seq_len, dim) 
        residue = x

        ## self-attention
        x = self.layernorm_1(x)
        x = self.attention(x)
        x += residue

        ## FFN 
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(x * 1.702) # Quick GeLU activation
        x = self.linear_2(x)
        x += residue
        return x 


class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module(
            # 12: number of head of the attention, 768 is the embedding size
            [CLIPLayer(12, 768)  for i in range(12)]
        )

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        #v(batch_size, seq_len) -> (batch_size, seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        # (batch_size, seq_len, dim)
        output = self.layernorm(state)




