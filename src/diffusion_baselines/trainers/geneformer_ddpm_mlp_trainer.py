# trainers/geneformer_ddpm_mlp_trainer.py

from .base_trainer import BaseTrainer

# MODIFIED: Renamed class for clarity and consistency
class MLPDDPMMLPGeneFormerTrainer(BaseTrainer):
    """Trainer for GeneFormer->DDPM->MLP pipeline."""
    def compute_loss(self, adata_batch, *args):
        # BaseTrainer already moved adata_batch to the device
        return self.model(adata_batch)