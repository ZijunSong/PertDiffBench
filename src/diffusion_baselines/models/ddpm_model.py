# models/ddpm_model.py

# This code file constructs a U-Net backbone for a DDPM (Denoising Diffusion Probabilistic Model).
# Author: Zijun Song
# Date: 2025-04

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Swish(nn.Module):
    """
    Swish activation: x * sigmoid(x).
    Paper: "Searching for Activation Functions" (Ramachandran et al., 2017).
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    """
    Maps diffusion timesteps to a dense embedding vector.
    Uses sinusoidal embeddings followed by two linear projections with Swish nonlinearity.
    """
    def __init__(self, T, d_model, dim):
        """
        Args:
            T (int): Total number of diffusion timesteps.
            d_model (int): Dimension of the sinusoidal embedding (must be even, half for sin and half for cos).
            dim (int): Output embedding dimension after projection.

        Ensures d_model is even, builds fixed sinusoidal embeddings of shape [T, d_model],
        then projects them with an MLP to produce a [T, dim] time embedding.
        """
        # Ensure half of d_model for sin and cos each
        assert d_model % 2 == 0, "d_model must be even"
        super().__init__()

        # Create the base sinusoidal frequencies emb[i] = exp(- (i / d_model) * log(10000))
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000) 
        emb = torch.exp(-emb)  # shape: [d_model/2]

        # Positions t = 0, 1, ..., T-1
        pos = torch.arange(T).float()  # shape: [T]

        # Outer product to get [T, d_model/2]: each row is t * freq
        emb = pos[:, None] * emb[None, :]
        # Stack sin and cos, then flatten to [T, d_model]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)  

        # Build a small MLP to project to 'dim'
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # fixed positional encodings
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        """Initialize linear layers using Xavier uniform and zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        """
        :param t: LongTensor of timesteps, shape [batch_size]
        :return: Tensor of shape [batch_size, dim], the time embeddings
        """
        emb = self.timembedding(t)
        return emb

class DownSample(nn.Module):
    """
    Halves spatial resolution via a stride-2 convolution.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        # temb unused here, but included for API consistency
        x = self.main(x)
        return x


class UpSample(nn.Module):
    """
    Doubles spatial resolution via nearest-neighbor upsampling followed by a conv layer.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        # Scale feature map back up
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

class AttnBlock(nn.Module):
    """
    Self-attention over spatial positions for context-dependent feature mixing.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        # Project to query, key, value, and output channels
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        # Xavier init for all convs; small gain for output proj
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        # Normalize then project to q/k/v
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # Reshape for batched matmul: q [B, HW, C], k [B, C, HW]
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        # Scaled dot-product attention weights
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        # Attend to v: v [B, HW, C]
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)

        # Final linear projection and residual add
        h = self.proj(h)
        return x + h

class ResBlock(nn.Module):
    """
    A residual block with optional attention and timestep conditioning.
    """
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        # First conv block: Norm → Swish → Conv2d
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        # Linear projection of time embedding to match out_ch
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        # Second conv block with dropout
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        # Shortcut if channels change
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        # Optional self-attention
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        """Xavier init for all Conv and Linear layers, small gain for final conv."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        """
        :param x: input feature map [B, in_ch, H, W]
        :param temb: time embedding [B, tdim]
        :return: output [B, out_ch, H, W]
        """
        h = self.block1(x)
        # add time embedding (broadcast over spatial dims)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        # Residual and optional attention
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class UNet(nn.Module):
    """
    U-Net architecture conditioned on diffusion timestep embeddings.
    """
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        """
        :param T: total diffusion timesteps
        :param ch: base channel count
        :param ch_mult: list of channel multipliers per downsample stage
        :param attn: list of indices where to apply attention
        :param num_res_blocks: number of ResBlocks per level
        :param dropout: dropout rate in ResBlocks
        """
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        # Time embedding dimension
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # Initial conv head
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # Build downsampling path
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            # Add downsample except at last stage
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # Bottleneck blocks
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        # Build upsampling path
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            # one extra ResBlock to match skip connections
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        # Final output head
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        """Xavier init for head/tail convs, small gain for tail."""
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        """
        :param x: noised image [B, 3, H, W]
        :param t: timesteps [B]
        :return: predicted noise residual [B, 3, H, W]
        """
        # 1) compute time embeddings
        temb = self.time_embedding(t)
        
        # 2) downsample path with skip-connections
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        
        # 3) bottleneck
        for layer in self.middleblocks:
            h = layer(h, temb)
        
        # 4) upsample path, concatenating skips
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        # 5) final conv to predict noise
        h = self.tail(h)
        assert len(hs) == 0
        return h
    
if __name__ == '__main__':
    # Quick sanity check
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print("Output shape:", y.shape)

