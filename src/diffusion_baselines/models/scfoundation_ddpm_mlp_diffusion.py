# models/scfoundation_ddpm_mlp_diffusion.py

import torch
from torch import nn
from .base import DiffusionModel
# No more scFoundation utils needed!
from .mlp_ddpm_mlp_autoencoder import ScRNADecoder
from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .mlp_ddpm_mlp_diffusion import SinusoidalPosEmb


class TimeConditionalWrapper(nn.Module):
    # This class remains unchanged
    def __init__(self, core_net: nn.Module, time_dim: int, latent_dim: int):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.fc_t = nn.Linear(time_dim, latent_dim)
        self.net = core_net

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        te = self.fc_t(te)
        return self.net(z + te)

# Renamed to reflect its new purpose: it's now a general DDPM for embeddings
class LatentDiffusionModel(DiffusionModel):
    def __init__(self, cfg):
        T     = cfg.model.diffusion.T
        betas = torch.linspace(cfg.model.diffusion.beta_1,
                               cfg.model.diffusion.beta_T,
                               T)
        super().__init__(T, betas)

        # The model no longer needs to load scFoundation.
        # It only needs to know the dimensions.
        latent_dim = cfg.model.decoder.latent_dim
        hidden_dim = cfg.model.decoder.hidden_dim
        
        # 1) Build the core_net for the diffusion process
        core_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 2) Wrap it to support time-steps
        time_dim = cfg.model.diffusion.hidden_dim
        conditioned_net = TimeConditionalWrapper(core_net, time_dim, latent_dim)

        # 3) Initialize Trainer & Sampler
        self.trainer = GaussianDiffusionTrainer(
            model=conditioned_net,
            beta_1=cfg.model.diffusion.beta_1,
            beta_T=cfg.model.diffusion.beta_T,
            T=T,
            conditional=False
        )
        self.sampler = GaussianDiffusionSampler(
            model=conditioned_net,
            beta_1=cfg.model.diffusion.beta_1,
            beta_T=cfg.model.diffusion.beta_T,
            T=T
        )

        # 4) The Decoder remains for the final sampling step
        self.decoder = ScRNADecoder(
            latent_dim,
            cfg.model.decoder.output_dim,
            hidden_dim
        )

    def forward(self, z0: torch.Tensor) -> torch.Tensor:    
        if z0.dim() == 1:
                print(f"[WARN] Input tensor has only 1 dimension (shape: {z0.shape}). Unsqueezing to add a batch dimension of 1.")
                z0 = z0.unsqueeze(0) # 例如, shape [3072] -> [1, 3072]
            
        # The forward pass now directly receives the pre-computed embedding z0
        B = z0.shape[0]
        t = torch.zeros(B, dtype=torch.long, device=z0.device)
        return self.trainer(z0, t)

    @torch.no_grad()
    def sample(self, num_samples: int, device: str) -> torch.Tensor:
        # Sampling now starts from pure noise of the correct shape
        latent_dim = self.decoder.latent_dim
        z_t = torch.randn((num_samples, latent_dim), device=device)
        B = num_samples
        
        # Denoising loop
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            eps = self.trainer.model(z_t, t)
            mean = self.sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var = self.sampler.posterior_var[step]
            if step > 0:
                noise = torch.randn_like(z_t)
                z_t = mean + torch.sqrt(var) * noise
            else:
                z_t = mean
        
        # Decode the final latent state back to gene expression space
        return self.decoder(z_t).clamp(-1, 1)