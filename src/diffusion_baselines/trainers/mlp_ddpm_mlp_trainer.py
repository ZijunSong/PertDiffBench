# trainers/mlp_ddpm_mlp_trainer.py

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

class ScRNATrainer:
    """
    Trainer for encoder → DDPM → decoder on paired scRNA-seq.
    Each batch is (x0, x1): pre- and post-perturbation expression.
    """
    def __init__(self, model: torch.nn.Module, diffusion, optimizer, scheduler, data_loader: DataLoader, device: torch.device, cfg):
        """
        Args:
            model: MLPDDPMMLP instance to train.
            diffusion: GaussianDiffusionTrainer (also held inside the model; passed explicitly here).
            optimizer: e.g. AdamW.
            scheduler: Learning-rate scheduler.
            data_loader: Training DataLoader.
            device: torch device ('cuda' or 'cpu').
            cfg: OmegaConf config.
        """
        self.model = model
        self.diffusion = diffusion  # model.forward uses the model's own diffusion_trainer internally
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader = data_loader
        self.device = device
        self.cfg = cfg
        
        self.train_cfg = cfg.train
        self.save_dir = self.train_cfg.save_weight_dir
        
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            
        self.current_epoch = 0
        self.current_step = 0

    def compute_loss(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Compute DDPM training loss for one batch.

        Args:
            x0: Pre-perturbation expression [B, G].
            x1: Post-perturbation expression [B, G].

        Returns:
            Scalar loss from the model forward pass.
        """
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        return self.model(x0, x1)

    def train(self):
        """Run the full training loop."""
        print("Starting training...")
        self.model.train()

        for epoch in range(self.current_epoch, self.train_cfg.epoch):
            self.current_epoch = epoch
            progress_bar = tqdm(
                self.loader,
                desc=f"Epoch {epoch+1}/{self.train_cfg.epoch}",
            )

            for x0, x1 in progress_bar:
                self.optimizer.zero_grad()
                loss = self.compute_loss(x0, x1)
                loss.backward()
                self.optimizer.step()

                self.current_step += 1

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                })

            self.scheduler.step()

            if self.save_dir and (epoch + 1) % self.train_cfg.ckpt_save_interval == 0:
                self.save_checkpoint(epoch)

        print("Training finished.")

    def save_checkpoint(self, epoch: int):
        """Save checkpoint (model, optimizer, scheduler)."""
        ckpt_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}.pth")

        torch.save({
            'epoch': epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, ckpt_path)

        print(f"Checkpoint saved to: {ckpt_path}")


class ScRNATrainerDrugCond(ScRNATrainer):
    """Trainer for drug-conditioned MLP-DDPM-MLP (MOA task)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_drug_structure = self.cfg.model.get("use_drug_structure", False)

    def compute_loss(self, x0: torch.Tensor, x1: torch.Tensor,
                     drug_idx: torch.Tensor = None, dose: torch.Tensor = None,
                     drug_dose: torch.Tensor = None) -> torch.Tensor:
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        if self.use_drug_structure:
            drug_dose = drug_dose.to(self.device).float() if isinstance(drug_dose, torch.Tensor) else torch.tensor(drug_dose, dtype=torch.float32, device=self.device)
            return self.model(x0, x1, drug_dose=drug_dose)
        drug_idx = drug_idx.to(self.device).long() if isinstance(drug_idx, torch.Tensor) else torch.tensor(drug_idx, dtype=torch.long, device=self.device)
        dose = dose.to(self.device).float() if isinstance(dose, torch.Tensor) else torch.tensor(dose, dtype=torch.float32, device=self.device)
        return self.model(x0, x1, drug_idx=drug_idx, dose=dose)

    def train(self):
        print("Starting training (drug-conditioned)...")
        self.model.train()
        for epoch in range(self.current_epoch, self.train_cfg.epoch):
            self.current_epoch = epoch
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.train_cfg.epoch}")
            for batch in progress_bar:
                if self.use_drug_structure:
                    x0, x1, drug_dose = batch
                    self.optimizer.zero_grad()
                    loss = self.compute_loss(x0, x1, drug_dose=drug_dose)
                else:
                    x0, x1, drug_idx, dose = batch
                    drug_idx = drug_idx if isinstance(drug_idx, torch.Tensor) else torch.tensor(drug_idx, dtype=torch.long)
                    dose = dose if isinstance(dose, torch.Tensor) else torch.tensor(dose, dtype=torch.float32)
                    self.optimizer.zero_grad()
                    loss = self.compute_loss(x0, x1, drug_idx=drug_idx, dose=dose)
                loss.backward()
                self.optimizer.step()
                self.current_step += 1
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                })
            self.scheduler.step()
            if self.save_dir and (epoch + 1) % self.train_cfg.ckpt_save_interval == 0:
                self.save_checkpoint(epoch)
        print("Training finished.")