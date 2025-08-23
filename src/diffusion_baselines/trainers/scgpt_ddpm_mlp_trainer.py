# trainers/mlp_ddpm_mlp_scgpt_trainer.py

from .base_trainer import BaseTrainer

class MLPDDPMMLPscGPTTrainer(BaseTrainer):
    """Trainer for scGPT→DDPM→MLP pipeline."""
    def compute_loss(self, adata_batch, *args):
        # BaseTrainer 已将 adata_batch 转到 device
        return self.model(adata_batch)
