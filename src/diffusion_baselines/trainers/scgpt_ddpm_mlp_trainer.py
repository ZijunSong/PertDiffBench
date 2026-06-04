# trainers/mlp_ddpm_mlp_scgpt_trainer.py

from .base_trainer import BaseTrainer

class MLPDDPMMLPscGPTTrainer(BaseTrainer):
    """Trainer for scGPT→DDPM→MLP pipeline."""
    def compute_loss(self, adata_batch, *args):
        # BaseTrainer already moves adata_batch to device
        return self.model(adata_batch)
