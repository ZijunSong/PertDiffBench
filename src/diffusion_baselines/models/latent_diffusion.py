# models/latent_diffusion.py

import torch
import torch.nn as nn
from .base import DiffusionModel
from .vae import VAE               # 假设 VAE 类里有 .encoder/.decoder 属性
from .ddpm_model import UNet       # 使用同一套 U-Net 结构
from .gaussian_diffusion import (
    GaussianDiffusionTrainer,
    GaussianDiffusionSampler
)

class LatentDiffusion(DiffusionModel):
    """
    Latent Diffusion Model:
    1) Encode input x into latent space z₀ via a VAE encoder.
    2) Add noise in z-space and train a U-Net to predict that noise.
    3) In sampling, start from pure noise z_T, iteratively denoise to z₀, then decode to x.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: config object with attributes
                - cfg.model.vae     for VAE hyperparameters
                - cfg.model.unet    for U-Net hyperparameters
                - cfg.model.beta_1, cfg.model.beta_T
        """
        super().__init__()

        # 1) Build VAE encoder & decoder
        #    Assume VAE(cfg.model.vae) 返回一个有 .encoder/.decoder 属性的实例
        vae_cfg = cfg.model.vae
        self.vae = VAE(vae_cfg)
        self.encoder = self.vae.encoder
        self.decoder = self.vae.decoder

        # 2) Build U-Net backbone operating in latent dimension
        unet_cfg = cfg.model.unet
        self.unet = UNet(
            T=unet_cfg.T,
            ch=unet_cfg.ch,
            ch_mult=unet_cfg.ch_mult,
            attn=unet_cfg.attn,
            num_res_blocks=unet_cfg.num_res_blocks,
            dropout=unet_cfg.dropout
        )

        # 3) Create diffusion trainer & sampler in latent space
        beta1 = cfg.model.beta_1
        betaT = cfg.model.beta_T
        T = unet_cfg.T

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

    def forward(self, x):
        """
        :param x: clean data batch, shape [B, C, H, W] or [B, D] for non-image
        :return: scalar loss for this batch
        """
        # 1) encode to latent z₀
        z0 = self.encoder(x)
        # 2) compute diffusion loss in latent space
        return self.trainer(z0)

    def sample(self, x_shape, device):
        """
        :param x_shape: tuple, shape of latent tensor to sample, e.g. (B, latent_dim, H', W')
        :param device: torch.device
        :return: reconstructed samples in data space [B, C, H, W]
        """
        # 1) draw pure noise in latent space
        zT = torch.randn(x_shape, device=device)
        # 2) run reverse diffusion to get z₀
        z0 = self.sampler(zT)
        # 3) decode to data space
        x_recon = self.decoder(z0)
        return x_recon
