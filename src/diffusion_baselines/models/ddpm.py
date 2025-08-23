# models/ddpm.py

import torch
import torch.nn as nn
from .base import DiffusionModel        # abstract base class
from .ddpm_model import UNet           # the U-Net backbone you wrote
from .gaussian_diffusion import (      # training & sampling logic
    GaussianDiffusionTrainer,
    GaussianDiffusionSampler,
)

class DDPM(DiffusionModel):
    """
    DDPM wrapper: holds a U-Net, a trainer (for computing loss)
    and a sampler (for generating samples).
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: a configuration object with attributes
                - cfg.model.unet.{T, ch, ch_mult, attn, num_res_blocks, dropout}
                - cfg.model.beta_1, cfg.model.beta_T
        """
        # 1) Extract schedule hyperparameters
        unet_cfg = cfg.model.unet
        T = unet_cfg.T
        beta1 = cfg.model.beta_1
        betaT = cfg.model.beta_T

        # 2) Build the beta schedule tensor
        betas = torch.linspace(beta1, betaT, T, dtype=torch.float32)

        # 3) Call base constructor with T and betas
        super().__init__(T, betas)

        # 4) Build the U-Net backbone
        self.unet = UNet(
            T=T,
            ch=unet_cfg.ch,
            ch_mult=unet_cfg.ch_mult,
            attn=unet_cfg.attn,
            num_res_blocks=unet_cfg.num_res_blocks,
            dropout=unet_cfg.dropout
        )

        # 5) Create trainer and sampler using the same noise schedule
        self.trainer = GaussianDiffusionTrainer(
            model=self.unet,
            beta_1=beta1,
            beta_T=betaT,
            T=T
        )
        self.sampler = GaussianDiffusionSampler(
            model=self.unet,
            beta_1=beta1,
            beta_T=betaT,
            T=T
        )

    def forward(self, x0):
        """
        :param x0: clean data batch, e.g. [B, C, H, W]
        :return: scalar loss (mean MSE over batch)
        """
        return self.trainer(x0)

    def sample(self, noise):
        """
        :param noise: initial noise tensor [B, C, H, W]
        :return: generated samples [B, C, H, W]
        """
        return self.sampler(noise)

# models/scrna_ddpm.py

from .scRNA_model import ScRNADDPM
from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler

class ScrnaDDPM(ScRNADDPM):
    def __init__(self, cfg):
        T      = cfg.model.unet.T
        betas  = torch.linspace(cfg.model.beta_1, cfg.model.beta_T, T)
        input_dim  = cfg.model.input_dim        # 例如 2000
        hidden_dim = cfg.model.hidden_dim       # 在 cfg 里指定
        super().__init__(T, betas, input_dim, hidden_dim)

        # Trainer & Sampler
        self.trainer = GaussianDiffusionTrainer(
            model=self, beta_1=cfg.model.beta_1,
            beta_T=cfg.model.beta_T, T=T
        )
        self.sampler = GaussianDiffusionSampler(
            model=self, beta_1=cfg.model.beta_1,
            beta_T=cfg.model.beta_T, T=T
        )

    def forward(self, x0):
        return self.trainer(x0)

    def sample(self, noise):
        return self.sampler(noise)
