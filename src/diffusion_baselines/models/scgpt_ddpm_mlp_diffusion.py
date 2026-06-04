# models/scgpt_ddpm_mlp_diffusion.py

import numpy as np
import scanpy as sc
import torch
from torch import nn
from .base import DiffusionModel
from utils.scgpt_utils import load_scgpt, embed_cells
from .mlp_ddpm_mlp_autoencoder import ScRNADecoder
from .gaussian_diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .mlp_ddpm_mlp_diffusion import SinusoidalPosEmb

class TimeConditionalWrapper(nn.Module):
    """Add timestep embedding to core_net so its signature is forward(z, t)."""
    def __init__(self, core_net: nn.Module, time_dim: int, latent_dim: int):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.fc_t = nn.Linear(time_dim, latent_dim)
        self.net = core_net

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 1) build timestep embedding
        te = self.time_emb(t)      # [B, time_dim]
        te = self.fc_t(te)         # [B, latent_dim]
        # 2) add embedding to z, then pass through core_net
        return self.net(z + te)


class MLPDDPMMLPscGPT(DiffusionModel):
    def __init__(self, cfg):
        T     = cfg.model.diffusion.T
        betas = torch.linspace(cfg.model.diffusion.beta_1,
                               cfg.model.diffusion.beta_T,
                               T)
        super().__init__(T, betas)

        # 1) scGPT load 
        device = torch.device(cfg.train.device)
        self.scgpt_model, self.scgpt_tokenizer = load_scgpt(cfg, device)

        # -- register forward hooks for debugging --
        def stats_hook(name):
            def hook(module, inp, out):
                tensor = out if isinstance(out, torch.Tensor) else out[0]
                if torch.isnan(tensor).any():
                    print(f"[NAN HUNT] NaN in {name} output! min/max = {tensor.min().item()}/{tensor.max().item()}")
                else:
                    print(f"[OK] {name} output OK, min/max = {tensor.min().item()}/{tensor.max().item()}")
            return hook

        for n, m in self.scgpt_model.named_modules():
            if isinstance(m, (torch.nn.LayerNorm, torch.nn.GELU, torch.nn.Linear)):
                m.register_forward_hook(stats_hook(n))
        # -- hooks registered --
        
        adata_ref = sc.read_h5ad(cfg.data.path)
        self.gene_ids = adata_ref.var_names.to_list()

        # 2) build core_net: reconstruct latent_dim only
        latent_dim = cfg.model.decoder.latent_dim
        hidden_dim = cfg.model.decoder.hidden_dim
        core_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 3) wrap with timestep-aware wrapper
        time_dim = cfg.model.diffusion.hidden_dim
        conditioned_net = TimeConditionalWrapper(core_net, time_dim, latent_dim)

        # 4) Trainer & Sampler (unconditional path also calls conditioned_net(z, t))
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

        # 5) decoder 
        self.decoder = ScRNADecoder(
            latent_dim,
            cfg.model.decoder.output_dim,
            hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) embed with scGPT to get z0
        z0 = embed_cells(
            self.scgpt_model, self.scgpt_tokenizer,
            x, self.gene_ids
        ).float()  # [B, latent_dim]
        # 2) use all-zero timesteps
        B = z0.shape[0]
        t = torch.zeros(B, dtype=torch.long, device=z0.device)
        # 3) pass to trainer
        return self.trainer(z0, t)

    @torch.no_grad()
    def sample(self, adata_ref):
        # 1) same as forward: get control embedding
        z0 = embed_cells(
            self.scgpt_model, self.scgpt_tokenizer,
            adata_ref.X, self.gene_ids
        ).float().to(self.betas.device)
        # 2) initialize noise
        z_t = torch.randn_like(z0)
        B = z0.shape[0]
        # 3) iter sample
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, dtype=torch.long, device=z0.device)
            eps = self.trainer.model(z_t, t)
            mean = self.sampler.predict_xt_prev_mean_from_eps(z_t, t, eps)
            var = self.sampler.posterior_var[step]
            if step > 0:
                z_t = mean + torch.sqrt(var) * torch.randn_like(z_t)
            else:
                z_t = mean
        # 4) decode
        return self.decoder(z_t).clamp(-1, 1)
