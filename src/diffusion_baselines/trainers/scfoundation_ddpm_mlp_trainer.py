# trainers/scfoundation_ddpm_mlp_trainer.py

from .base_trainer import BaseTrainer

# Renamed the class to reflect the new model
class LatentDDPMTrainer(BaseTrainer):
    """Trainer for scFoundation->DDPM->MLP pipeline."""
    def compute_loss(self, adata_batch, *args):
        # BaseTrainer already moves adata_batch to the device
        return self.model(adata_batch)