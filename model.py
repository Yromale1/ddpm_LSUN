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
        return emb

class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.mlp(self.time_embed(t))

class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        gamma, beta = self.fc(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(16, in_channels)
        self.norm2 = nn.GroupNorm(16, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.film = FiLM(cond_dim, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        residual = self.res_conv(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.film(x, cond)

        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv2(x)

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
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        q = self.query(x_flat).permute(0, 2, 1)
        k = self.key(x_flat)
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        v = self.value(x_flat)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, use_attention=False):
        super().__init__()
        self.resblock = ResidualBlock(in_c, out_c, cond_dim)
        self.resblock2 = ResidualBlock(out_c, out_c, cond_dim)
        self.attention = SelfAttention(out_c) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_c, out_c, 4, 2, 1)

    def forward(self, x, cond):
        x = self.resblock(x, cond)
        x = self.resblock2(x, cond)
        x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, cond_dim, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.resblock = ResidualBlock(in_c + skip_c, out_c, cond_dim)
        self.resblock2 = ResidualBlock(out_c, out_c,cond_dim)
        self.attention = SelfAttention(out_c) if use_attention else nn.Identity()

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x, cond)
        x = self.resblock2(x, cond)
        x = self.attention(x)
        return x

class ConditionalUNet(nn.Module):
    def __init__(self, input_c=3, base_c=128, cond_dim=128, n_classes=5):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, cond_dim)
        self.time_embed = TimeMLP(cond_dim)

        self.init_conv = nn.Conv2d(input_c, base_c, 3, padding=1)

        self.down1 = DownBlock(base_c, base_c, cond_dim)           # 128 → 64
        self.down2 = DownBlock(base_c, base_c * 2, cond_dim)       # 64 → 32
        self.down3 = DownBlock(base_c * 2, base_c * 4, cond_dim, use_attention=True)  # 32 → 16
        self.down4 = DownBlock(base_c * 4, base_c * 8, cond_dim, use_attention=True)  # 16 → 8
        self.down5 = DownBlock(base_c * 8, base_c * 16, cond_dim)  # 8 → 4

        self.bottleneck = ResidualBlock(base_c * 16, base_c * 16, cond_dim)
        self.bottleneck2 = ResidualBlock(base_c * 16, base_c * 16, cond_dim)
        self.attn_bottleneck = SelfAttention(base_c * 16)

        self.up0 = UpBlock(base_c * 16, base_c * 8, base_c * 8, cond_dim)
        self.up1 = UpBlock(base_c * 8, base_c * 4, base_c * 4, cond_dim, use_attention=True)
        self.up2 = UpBlock(base_c * 4, base_c * 2, base_c * 2, cond_dim, use_attention=True)
        self.up3 = UpBlock(base_c * 2, base_c, base_c, cond_dim)
        self.up4 = UpBlock(base_c, base_c, base_c, cond_dim)

        self.final_conv = nn.Conv2d(base_c, input_c, 1)

    def forward(self, x, t, y):
        time_emb = self.time_embed(t)
        if y is not None:
            label_emb = self.label_embed(y)
        else:
            label_emb = torch.zeros_like(time_emb)
        cond = label_emb + time_emb
        x = self.init_conv(x)

        x, skip1 = self.down1(x, cond)
        x, skip2 = self.down2(x, cond)
        x, skip3 = self.down3(x, cond)
        x, skip4 = self.down4(x, cond)
        x, skip5 = self.down5(x, cond)

        x = self.bottleneck(x, cond)
        x = self.bottleneck2(x, cond)
        x = self.attn_bottleneck(x)

        x = self.up0(x, skip5, cond)
        x = self.up1(x, skip4, cond)
        x = self.up2(x, skip3, cond)
        x = self.up3(x, skip2, cond)
        x = self.up4(x, skip1, cond)

        return self.final_conv(x)
