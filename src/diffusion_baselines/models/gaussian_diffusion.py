# models/gaussian_diffusion.py

# This code file is used to implement the theoretical derivation part of the diffusion model DDPM.
# Author: Zijun Song
# Date: 2025-04

import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(v, t, x_shape):
    """
    Extract values from a 1-D tensor at specified timesteps and reshape for broadcasting.

    Args:
        v (Tensor): A 1-D tensor of length T, containing precomputed coefficients.
        t (LongTensor): Tensor of shape [batch_size], each entry is an integer timestep in [0, T).
        x_shape (tuple): The shape of the target tensor x_t, e.g. [batch_size, channels, height, width].

    Returns:
        Tensor: A tensor of shape [batch_size, 1, 1, ..., 1] that can broadcast over x_t.
    """
    device = t.device
    # Gather the coefficients for each sample's timestep and cast to float
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # # Reshape to [batch_size] + [1]*(len(x_shape)-1)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def extract(buf, t, shape):
    """
    从一维张量 buf 中按照索引 t 取值，并 reshape 成 shape。
    """
    out = buf.gather(-1, t)            # [B]
    return out.view([t.shape[0]] + [1]*(len(shape)-1)).expand(shape)

class GaussianDiffusionTrainer(nn.Module):
    """
    Trainer for (conditional) DDPM. 支持条件 diffusion。
    """
    def __init__(self, model: nn.Module, beta_1: float, beta_T: float, T: int, conditional: bool=False):
        super().__init__()
        self.model = model
        self.T = T
        self.conditional = conditional

        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        self.register_buffer('betas', betas)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0: torch.Tensor, cond: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            x_0: 目标干净样本（latent 或者原始），形状 [B, D]
            cond: 可选的条件向量，形状 [B, D_cond]
        Returns:
            标量 loss
        """
        B = x_0.shape[0]
        # 1) 随机选 t
        t = torch.randint(0, self.T, (B,), device=x_0.device, dtype=torch.long)
        # 2) 采噪声
        noise = torch.randn_like(x_0)
        # 3) 构造 x_t
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )
        # 4) 调用网络预测噪声
        if self.conditional:
            assert cond is not None, "Conditional trainer requires cond tensor"
            pred = self.model(x_t, cond, t)
        else:
            pred = self.model(x_t, t)
        # 5) 计算 MSE
        loss = F.mse_loss(pred, noise, reduction='none')
        return loss.mean()

class GaussianDiffusionSampler(nn.Module):
    """
    Sampler module for DDPM. Implements the reverse diffusion pθ(x_{t−1} | x_t) to generate samples from pure noise.
    """
    def __init__(self, model, beta_1, beta_T, T):
        """
        Initialize reverse diffusion constants.

        Args:
            model (nn.Module): The trained denoising network εθ(x_t, t).
            beta_1 (float): Initial β₁ in the variance schedule.
            beta_T (float): Final β_T in the variance schedule.
            T (int): Number of diffusion steps.
        """
        super().__init__()
        self.model = model
        self.T = T

        # Same beta schedule and α_t, ᾱ_t as in trainer
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # ᾱ_{t−1}, with ᾱ_{−1}=1 for padding
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # Coefficients for predicting x_{t−1} mean from x_t and ε
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # Variance of the posterior q(x_{t−1} | x_t, x_0)
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        Compute the predicted mean of pθ(x_{t−1} | x_t) given εθ.

        Args:
            x_t (Tensor): Current noisy data.
            t (LongTensor): Current timestep indices.
            eps (Tensor): Noise predictions from the model εθ(x_t, t).

        Returns:
            Tensor: Predicted mean of x_{t−1}.
        """
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        """
        Compute both mean and variance for pθ(x_{t−1} | x_t).

        Args:
            x_t (Tensor): Noisy data at timestep t.
            t (LongTensor): Current timestep indices.

        Returns:
            Tuple[Tensor, Tensor]: (mean, variance) for sampling x_{t−1}.
        """
        # Use β₂...β_T for variance schedule, but keep β₁ as special case
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        # Predict noise via network
        eps = self.model(x_t, t)
        # Compute predicted mean via closed-form
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Generate a sample from pure noise by iterating Algorithm 2.

        Args:
            x_T (Tensor): Initial noise, shape [B, C, H, W].

        Returns:
            Tensor: Final denoised sample clipped to [-1,1].
        """
        x_t = x_T
        # Loop backwards from T-1 to 0
        for time_step in reversed(range(self.T)):
            # Create a tensor filled with current timestep index
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # Get predicted mean and variance of x_{t−1}
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # Add noise for all but the last step
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            # Sample x_{t−1} ~ N(mean, var)
            x_t = mean + torch.sqrt(var) * noise
            # Sanity check for NaNs
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        # Final output
        x_0 = x_t
        return torch.clip(x_0, -1, 1)  
