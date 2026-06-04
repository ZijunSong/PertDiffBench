# src/diffusion_baselines/models/scimilarity_latent_ddpm_mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .mlp_ddpm_mlp_diffusion import MLPCond  # reuse existing conditional MLP


class ScimilarityLatentDDPMMLP(nn.Module):
    """
    DDPM in SCimilarity latent space + MLP decoder back to gene space.

    traininput:
        z0: control latent  [B, latent_dim]
        z1: perturbed latent [B, latent_dim]
        x1: perturbed gene expression [B, G]

    compute_loss returns:
        loss_total, loss_diff, loss_dec
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        ae_cfg = cfg.model.ae
        diff_cfg = cfg.model.diffusion

        latent_dim = ae_cfg.latent_dim  # set automatically by train script to X_scim dim
        gene_dim = ae_cfg.input_dim      # = adata.n_vars
        hidden_ae = ae_cfg.hidden_dim
        hidden_diff = diff_cfg.hidden_dim

        # decoder: latent -> gene
        from .mlp_ddpm_mlp_autoencoder import ScRNADecoder
        self.decoder = ScRNADecoder(
            latent_dim,
            gene_dim,
            hidden_ae,
        )

        # noise predictor in latent space
        self.net = MLPCond(
            latent_dim=latent_dim,
            hidden_dim=hidden_diff,
            cond_dim=latent_dim,
            time_dim=hidden_diff,
        )

        beta1, betaT, T = diff_cfg.beta_1, diff_cfg.beta_T, diff_cfg.timesteps

        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.net,
            beta_1=beta1,
            beta_T=betaT,
            T=T,
            conditional=True,
        )
        self.diffusion_sampler = GaussianDiffusionSampler(
            model=self.net,
            beta_1=beta1,
            beta_T=betaT,
            T=T,
        )

        # decoder loss weight; set via model.dec_weight in config
        self.dec_weight = getattr(cfg.model, "dec_weight", 1.0)

    def compute_loss(self, z0: torch.Tensor, z1: torch.Tensor, x1: torch.Tensor):
        """
        z0: [B, latent_dim] control latent
        z1: [B, latent_dim] perturbed latent
        x1: [B, G]          perturbed gene expression
        """
        # 1) diffusion loss: learn z1 in latent space
        loss_diff = self.diffusion_trainer(z1, cond=z0)

        # 2) decoder loss: reconstruct x1 from clean z1
        x1_hat = self.decoder(z1)
        loss_dec = F.mse_loss(x1_hat, x1)

        loss_total = loss_diff + self.dec_weight * loss_dec
        return loss_total, loss_diff, loss_dec

    @torch.no_grad()
    def sample_from_latent(self, z0: torch.Tensor, noise: torch.Tensor = None):
        """
        Given control latent z0, run DDPM+decoder to predict gene expression.

        z0: [B, latent_dim]
        noise: optional initial noise, [B, latent_dim]

        Returns:
            x1_pred: [B, G]
        """
        device = z0.device
        B, latent_dim = z0.shape

        if noise is not None:
            z_t = noise.to(device)
        else:
            z_t = torch.randn_like(z0)

        T = self.diffusion_trainer.T

        for step in reversed(range(T)):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            eps = self.net(z_t, z0, t)
            mean = self.diffusion_sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var = self.diffusion_sampler.posterior_var[step]

            if step > 0:
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = mean

        # decode z_t -> gene space
        if hasattr(self.decoder, "net") and isinstance(self.decoder.net[0], nn.Linear):
            target_dtype = self.decoder.net[0].weight.dtype
            z_t = z_t.to(dtype=target_dtype)
        else:
            z_t = z_t.float()

        x1_pred = self.decoder(z_t)
        return x1_pred
