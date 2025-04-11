import torch 
from torch import nn
from torch.nn import functional as F   
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, n_embd * 4)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # (1, 1280) -> (1, 320)
        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels) 
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels) 
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) 
        
    def forward(sel, feature, time):
        # feature: (batch_size, in_channels, height, width)
        # time: (1, 1280)
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd 
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=3, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels * 2, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
    
    def forward(self, x, context):
        # (batch_size, features, height, width)
        # context: (batch_size, seq_len, dim)

        residue_long = x 
        x = self.groupnorm(x)
        x = self.conv_input

        n, c, h, w = x.shape

        # (batch_size, features, height * width)
        x = x.view(n, c, h * w) 

        # (batch_size, height * width. features)
        x = x.transpose(-1, -2)

        # Normalization + self attention with skip connection
        residue_short = x 
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short
        residue_short = x 

        # Normalization + cross attention with skip connection
        x = self.layernorm_2(x)
        # Cross Attention
        self.attention_2(x, context)

        x += residue_short 
        residue_short = x

        # Normalization + FF with GeGLU and skip connection

        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residue_short 
        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)
        return self.conv_output(x) + residue_long





class  SwithSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttetionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super_().__init__()
        self.encoder = nn.Module([
            # (batch_size, 4, Height / 8, Width / 8)
            SwithSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwithSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwithSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (batch_size, 320, Height / 8, Width / 8) -> (batch_size, 320, Height / 16, Width / 16)
            SwithSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwithSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwithSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (batch_size, 640, Height / 16, Width / 16) -> (batch_size, 640, Height / 32, Width / 32)
            SwithSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwithSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwithSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            # (batch_size, 1280, Height / 32, Width / 32) -> (batch_size, 1290, Height / 64, Width / 64)
            SwithSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwithSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwithSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
        ])

    self.bottleneck = SwithSequential(
        UNET_ResidualBlock(1280, 1280),
        UNET_AttentionBlock(8, 160),
        UNET_ResidualBlock(1280, 1280),
    )

    self.decoders = nn.ModuleList([
        # (batch_size, 2560, Height / 64, Width / 64) -> (batch_size, 1280, Height / 64, Width / 64)
        SwithSequential(UNET_ResidualBlock(2560, 1280)),
        SwithSequential(UNET_ResidualBlock(2560, 1280)),
        SwithSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

        SwithSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
        SwithSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
        SwithSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

        SwithSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
        SwithSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
        SwithSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), upSample(640)),

        SwithSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

        # Do skip-connection -> concat feature encode + decoder, giu nguyen 80 tu encoder
        SwithSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 80)),
        SwithSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 40)),
    ])


class UNET_OutputLayer(nn.Module):
    def _init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (batch_size, 320, height / 8, width / 8) 
        x = self.groupnorm(x)

        x = F.silu(x)

        # (batch_size, 4, height / 8, width / 8)
        x = self.conv(x)
        return x 



class Upsample(nn.Module):
    def __init__(sekf, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        #(batch_size, features, height, width) -> (batch_size, features, height * 2, width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class Diffusion(nn.Module):
    def __init_(self):
        self.time_embedding = TimeEmbedding(320, 4)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch_size, 4, h / 8, w / 8)
        # context: (batch_size, seq_len, dim)
        # time: (1, 320)

        # (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, h / 8, w / 8) -> (batch_size, 320, h / 8, w / 8)
        output = self.unet(latent, context, time)

        # (batch_size, 320, h / 8, w / 8) -> (batch_size, 4, h / 8, w / 8)
        output = self.final(output)
        return output
    