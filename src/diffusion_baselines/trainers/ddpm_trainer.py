# trainers/ddpm_trainer.py

from .base_trainer import BaseTrainer

class DDPMTrainer(BaseTrainer):
    """
    Trainer for DDPM models. Expects `self.model` to be an instance
    of a DDPM model whose forward(x) returns a scalar diffusion loss.
    """
    def compute_loss(self, x, *args):
        """
        Compute the mean-squared error diffusion loss for a batch.

        Args:
            x (Tensor): Clean data batch, shape [B, C, H, W].
            *args: Additional unused arguments (e.g. labels).

        Returns:
            Tensor: Scalar loss (averaged over batch).
        """
        # Move input to the same device as model
        x = x.to(self.device)
        # Model.forward returns the diffusion loss
        loss = self.model(x)
        return loss
