# trainers/scvi_ddpm_mlp_trainer.py

import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer

class ScviDdpmMlpTrainer(BaseTrainer):
    """
    Joint training of scVI encoder + DDPM + MLP decoder:
        - diffusion loss: learn z1|z0 in latent space
        - recon loss: learn z0 → x0 in gene space
    """
    def compute_loss(self, x0, x1):
        """
        x0: Control scRNA [B, G]
        x1: Perturbed scRNA [B, G]
        """
        # 1) encode both batches
        # scVI encoder + convert to latent
        z0 = self.model.encode_fn(x0)    # [B, L]
        z1 = self.model.encode_fn(x1)    # [B, L]

        # 2) diffusion loss in latent space
        diff_loss = self.model.ddpm.diffusion_trainer(z1, cond=z0)

        # 3) reconstruction loss for MLP decoder: z0 -> x0
        x0_pred = self.model.ddpm.decoder(z0)  # [B, G]
        recon_loss = F.mse_loss(x0_pred, x0)

        # 4) combine total loss
        w = self.cfg.train.recon_weight
        loss = diff_loss + w * recon_loss
        return loss
