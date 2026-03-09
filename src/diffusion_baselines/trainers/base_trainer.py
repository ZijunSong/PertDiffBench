# trainers/base_trainer.py

import os
import torch
from torch.optim import Adam
from tqdm import tqdm

class BaseTrainer:
    """
    BaseTrainer handles the common training loop:
      1) iteration over epochs and batches
      2) optimizer / scheduler stepping
      3) checkpoint saving and loading
    Subclasses must implement compute_loss() to return a scalar loss.
    """
    def __init__(self, model, optimizer, scheduler, loader, cfg, device=None):
        self.model     = model.to(device or cfg.train.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader    = loader
        self.cfg       = cfg
        self.device    = device or cfg.train.device

        # optionally resume from checkpoint
        if getattr(cfg.train, "resume_from", None):
            self._load_checkpoint(cfg.train.resume_from)

    def train(self):
        """
        Run the full training loop over epochs.
        """
        for epoch in range(1, self.cfg.train.epoch + 1):
            self.model.train()
            epoch_loss = 0.0

            # iterate batches
            for x, *rest in tqdm(self.loader, desc=f"Epoch {epoch}/{self.cfg.train.epoch}"):
                x = x.to(self.device)
                # if there are additional targets (e.g., labels), send them too
                rest = [r.to(self.device) for r in rest]

                self.optimizer.zero_grad()
                loss = self.compute_loss(x, *rest)
                loss.backward()

                # gradient clipping if configured
                if getattr(self.cfg.train, "grad_clip", None):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.train.grad_clip
                    )

                self.optimizer.step()
                epoch_loss += loss.item()

            # step scheduler after each epoch
            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / len(self.loader)
            print(f"[Epoch {epoch}] avg loss: {avg_loss:.4f}")

            # save checkpoint every ckpt_interval epochs (default: every epoch)
            interval = getattr(self.cfg.train, "ckpt_save_interval", 1)
            if epoch % interval == 0:
                self.save_checkpoint(epoch)

    def compute_loss(self, x, *rest):
        """
        Compute and return the scalar loss for a batch.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def save_checkpoint(self, epoch):
        """
        Save model and optimizer state.
        """
        ckpt_dir = self.cfg.train.save_weight_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"ckpt_{epoch}.pt")

        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "cfg": self.cfg,
        }, path)
        print(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path):
        """
        Load model and optimizer state from a checkpoint.
        """
        print(f"Loading checkpoint from {path} ...")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")
        return start_epoch
