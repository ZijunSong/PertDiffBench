# trainers/scvi_ddpm_mlp_trainer.py

import torch
import torch.nn.functional as F
from .base_trainer import BaseTrainer

class ScviDdpmMlpTrainer(BaseTrainer):
    """
    将 scVI encoder + DDPM + MLP decoder 联合训练：
      - diffusion loss: 在 latent 空间学习 z1|z0
      - recon loss:   在 latent 空间学习 z0 → x0
    """
    def compute_loss(self, x0, x1):
        """
        x0: Control scRNA [B, G]
        x1: Perturbed scRNA [B, G]
        """
        # 1) 获取各部分
        # scVI encoder + convert to latent
        z0 = self.model.encode_fn(x0)    # [B, L]
        z1 = self.model.encode_fn(x1)    # [B, L]

        # 2) diffusion loss in latent space
        diff_loss = self.model.ddpm.diffusion_trainer(z1, cond=z0)

        # 3) reconstruction loss for MLP decoder: z0 -> x0
        x0_pred = self.model.ddpm.decoder(z0)  # [B, G]
        recon_loss = F.mse_loss(x0_pred, x0)

        # 4) 合成总 loss
        w = self.cfg.train.recon_weight
        loss = diff_loss + w * recon_loss
        return loss
