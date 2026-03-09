# models/base.py

# This code file constructs a base Diffusion model.
# Author: Zijun Song
# Date: 2025-05

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DiffusionModel(nn.Module, ABC):
    """
    Base class for all diffusion models.

    Args:
        T (int): total number of diffusion timesteps.
        betas (torch.Tensor): noise schedule of shape (T,).
    """
    def __init__(self, T: int, betas: torch.Tensor):
        super().__init__()
        self.T = T
        self.betas = betas
        # Precompute alphas etc. if needed
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given noisy input x at timestep t, predict model output.

        Args:
            x (Tensor): shape (B, C, H, W) or (B, D) for scRNA.
            t (Tensor): shape (B,) containing integers in [0, T).
        Returns:
            Tensor: model prediction (e.g. noise estimate).
        """
        raise NotImplementedError

    def sample(self, shape, device: torch.device, **kwargs):
        """
        Generate new samples by running the reverse diffusion process.

        Args:
            shape: tuple describing output shape, e.g. (B, C, H, W)
            device: torch device
        Returns:
            Tensor: generated samples
        """
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            # call forward to predict noise or x0, then step
            eps = self.forward(x, torch.full((shape[0],), t, device=device, dtype=torch.long), **kwargs)
            # compute posterior mean & variance (not shown)
            # x = posterior_mean + noise * posterior_std
        return x

    def compute_loss(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Standard MSE loss between true noise and predicted noise.

        Args:
            x0: original clean data
            x_t: noisy data at timestep t
            t: timesteps
        """
        # q_sample samples noise eps_true, here omitted for brevity
        eps_pred = self.forward(x_t, t)
        eps_true = (x_t - torch.sqrt(self.alpha_cumprod[t])[:, None] * x0) \
                   / torch.sqrt(1 - self.alpha_cumprod[t])[:, None]
        return nn.functional.mse_loss(eps_pred, eps_true)
