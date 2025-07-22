import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -(np.log(10000) / half_dim))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # shape: (B, dim)
    
class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)  # gamma and beta

    def forward(self, x, cond):
        # cond: (batch, cond_dim)
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.film = FiLM(cond_dim, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        residual = self.res_conv(x)
        x = F.relu(self.conv1(x))
        x = self.film(x, cond)
        x = F.relu(self.conv2(x))
        x = self.film(x, cond)
        return x + residual

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, H, W) -> flatten spatial dims
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W)  # (B, C, N)
        q = self.query(x_flat).permute(0, 2, 1)  # (B, N, C//8)
        k = self.key(x_flat)  # (B, C//8, N)
        attn = torch.bmm(q, k)  # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        v = self.value(x_flat)  # (B, C, N)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, use_attention=False):
        super().__init__()
        self.resblock = ResidualBlock(in_c, out_c, cond_dim)
        self.attention = SelfAttention(out_c) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_c, out_c, 4, 2, 1)  # halve spatial size

    def forward(self, x, cond):
        x = self.resblock(x, cond)
        x = self.attention(x)
        skip = x  # skip connection avant downsample
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, cond_dim, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.resblock = ResidualBlock(out_c + skip_c, out_c, cond_dim)
        self.attention = SelfAttention(out_c) if use_attention else nn.Identity()

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x, cond)
        x = self.attention(x)
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, input_c=3, base_c=64, cond_dim=128, n_classes=5):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, cond_dim)
        self.time_embed = SinusoidalTimeEmbedding(cond_dim)

        self.init_conv = nn.Conv2d(input_c, base_c, 3, padding=1)

        self.down1 = DownBlock(base_c, base_c*2, cond_dim)
        self.down2 = DownBlock(base_c*2, base_c*4, cond_dim, use_attention=True)
        self.down3 = DownBlock(base_c*4, base_c*8, cond_dim)

        self.bottleneck = ResidualBlock(base_c*8, base_c*8, cond_dim)
        self.attn_bottleneck = SelfAttention(base_c*8)

        self.up3 = UpBlock(base_c*8, base_c*8, base_c*4, cond_dim)
        self.up2 = UpBlock(base_c*4, base_c*4, base_c*2, cond_dim, use_attention=True)
        self.up1 = UpBlock(base_c*2, base_c*2, base_c, cond_dim)

        self.final_conv = nn.Conv2d(base_c, input_c, 1)

    def forward(self, x, t, y):
        cond = self.label_embed(y) + self.time_embed(t)
        x = self.init_conv(x)

        d1, skip1 = self.down1(x, cond)
        d2, skip2 = self.down2(d1, cond)
        d3, skip3 = self.down3(d2, cond)

        b = self.bottleneck(d3, cond)
        b = self.attn_bottleneck(b)

        u3 = self.up3(b, skip3, cond)
        u2 = self.up2(u3, skip2, cond)
        u1 = self.up1(u2, skip1, cond)

        return self.final_conv(u1)