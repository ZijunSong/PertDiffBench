# models/mlp_ddpm_mlp_diffusion.py

import torch
import torch.nn as nn
from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler

# Utility to extract buffer values
def extract(buf, t, shape):
    out = buf.gather(-1, t)
    return out.view([t.shape[0]] + [1] * (len(shape) - 1)).expand(shape)

class SinusoidalPosEmb(nn.Module):
    """时序步长的正余弦嵌入，输出 float32。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = timesteps.device
        # 强制 float32
        exp_term = -torch.log(torch.tensor(10000.0, dtype=torch.float32, device=device))
        omega = torch.exp(
            exp_term * torch.arange(half_dim, device=device, dtype=torch.float32) / (half_dim - 1)
        )
        args = timesteps[:, None].float() * omega[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class MLPCond(nn.Module):
    """带条件和时间嵌入的 MLP 噪声预测网络。"""
    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, time_dim: int):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.fc_t     = nn.Linear(time_dim, hidden_dim)
        self.fc1      = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.act      = nn.SiLU()
        self.fc2      = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1) 输入统一为 float32
        target_dtype = self.fc1.weight.dtype
        z    = z.to(dtype=target_dtype)
        cond = cond.to(dtype=target_dtype)
        # 2) 时间嵌入
        te = self.time_emb(t)
        te = self.act(self.fc_t(te))
        # 3) 拼接与前向
        h = torch.cat([z, cond], dim=-1)
        h = self.act(self.fc1(h))
        h = h + te
        return self.fc2(h)

class MLPDDPMMLP(nn.Module):
    """Encoder→DDPM→Decoder 带条件扩散整体模型。"""
    def __init__(self, cfg):
        super().__init__()
        # Autoencoder
        from .mlp_ddpm_mlp_autoencoder import ScRNAEncoder, ScRNADecoder
        ae = cfg.model.ae
        self.encoder = ScRNAEncoder(ae.input_dim, ae.latent_dim, ae.hidden_dim)
        self.decoder = ScRNADecoder(ae.latent_dim, ae.input_dim, ae.hidden_dim)
        # Diffusion schedule
        diff = cfg.model.diffusion
        beta1, betaT, T = diff.beta_1, diff.beta_T, diff.timesteps
        # Noise predictor
        self.net = MLPCond(
            latent_dim=ae.latent_dim,
            hidden_dim=diff.hidden_dim,
            cond_dim=ae.latent_dim,
            time_dim=diff.hidden_dim
        )
        # Conditional trainer & sampler
        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.net, beta_1=beta1, beta_T=betaT, T=T, conditional=True
        )
        self.diffusion_sampler = GaussianDiffusionSampler(
            model=self.net, beta_1=beta1, beta_T=betaT, T=T
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        z0 = self.encoder(x0)
        z1 = self.encoder(x1)
        return self.diffusion_trainer(z1, cond=z0)

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        device = x0.device
        z0 = self.encoder(x0)
        # 初始噪声
        z_t = noise.to(device) if noise is not None else torch.randn_like(z0)
        B = z0.shape[0]
        for step in reversed(range(self.diffusion_trainer.T)):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            eps  = self.net(z_t, z0, t)
            mean = self.diffusion_sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var  = self.diffusion_sampler.posterior_var[step]
            if step > 0:
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = mean
        # 确保 float32，再解码
        z_t = z_t.to(dtype=self.decoder.net[0].weight.dtype) if hasattr(self.decoder, 'net') else z_t.float()
        x1 = self.decoder(z_t)
        return x1.clamp(-1, 1)
