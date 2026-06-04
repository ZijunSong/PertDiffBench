#!/usr/bin/env python3

import os
import argparse
from glob import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
import scanpy as sc

from src.diffusion_baselines.models.scvi_latent_ddpm_mlp import ScviLatentDDPMMLP
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler


class PairedLatentGeneDataset(Dataset):
    """
    Generic latent+gene dataset.

    Returns:
        z0: control latent
        z1: perturbed latent
        x1: perturbed gene expression (gene space)

    Assumptions:
        - adata.X: gene expression (sparse or dense)
        - adata.obsm[latent_key]: latent representation
        - adata.obs["perturbation_status"]: 'Control' or perturbation labels
    """

    def __init__(self, adata, latent_key: str = "X_scvi"):
        super().__init__()
        self.adata = adata
        self.latent_key = latent_key

        if "perturbation_status" not in adata.obs.columns:
            raise KeyError("adata.obs must contain 'perturbation_status'.")

        if latent_key not in adata.obsm:
            raise KeyError(
                f"adata.obsm['{latent_key}'] not found. "
                f"Please run your encoder (e.g., scVI, scGPT) first."
            )

        self.latent = adata.obsm[latent_key]

        self.ctrl_mask = adata.obs["perturbation_status"] == "Control"
        self.ctrl_ids = np.where(self.ctrl_mask.values)[0]

        self.pert_indices = np.where(~self.ctrl_mask.values)[0]
        self.pert_labels = adata.obs["perturbation_status"].values[self.pert_indices]

        # group perturbed cells by label for possible smarter sampling later
        self.pert_groups = defaultdict(list)
        for idx, label in zip(self.pert_indices, self.pert_labels):
            self.pert_groups[label].append(idx)
        self.pert_labels_unique = list(self.pert_groups.keys())

        print(f"[Dataset] #control cells: {len(self.ctrl_ids)}")
        print(f"[Dataset] #perturbation groups: {len(self.pert_labels_unique)}")
        print(f"[Dataset] Using latent_key='{latent_key}' with shape {self.latent.shape}")

    def __len__(self):
        # use number of perturbed cells as dataset size
        return len(self.pert_indices)

    def __getitem__(self, idx):
        # pick one perturbed cell
        pert_cell_idx = self.pert_indices[idx]
        pert_label = self.pert_labels[idx]  # currently unused, but kept for future extension

        # random control cell
        ctrl_idx = np.random.choice(self.ctrl_ids)

        X = self.adata.X
        z_all = self.latent

        # control latent
        z0 = z_all[ctrl_idx]
        # perturbed latent & gene
        z1 = z_all[pert_cell_idx]
        x1 = X[pert_cell_idx].toarray() if hasattr(X, "toarray") else X[pert_cell_idx]

        z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        z1 = torch.from_numpy(np.asarray(z1, dtype=np.float32))
        x1 = torch.from_numpy(np.asarray(x1, dtype=np.float32))

        return z0, z1, x1


def find_latest_checkpoint(save_dir: str):
    """
    Find the latest checkpoint in a directory.
    Priority:
        1) model_final.pth
        2) highest epoch model_epoch_XXX.pth
    """
    final_path = os.path.join(save_dir, "model_final.pth")
    if os.path.exists(final_path):
        return final_path, True  # is_final=True

    pattern = os.path.join(save_dir, "model_epoch_*.pth")
    ckpts = glob(pattern)
    if not ckpts:
        return None, False

    def _epoch_num(path):
        basename = os.path.basename(path)
        # expected: model_epoch_100.pth
        try:
            num = basename.replace("model_epoch_", "").replace(".pth", "")
            return int(num)
        except Exception:
            return -1

    ckpts = sorted(ckpts, key=_epoch_num)
    return ckpts[-1], False  # latest, but not final


def main():
    parser = argparse.ArgumentParser(
        description="Train DDPM-MLP in generic latent space (scVI/scGPT/etc.), "
                    "with decoder back to gene space, and resume from checkpoints."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config for ScviLatentDDPMMLP (or compatible model).",
    )
    parser.add_argument(
        "--train-data-path",
        required=True,
        help="Path to training AnnData (.h5ad) that already has obsm[latent_key].",
    )
    parser.add_argument(
        "--latent-key",
        type=str,
        default="X_scvi",
        help="Key in adata.obsm that stores latent representation, "
             "e.g. 'X_scvi', 'X_scgpt', etc.",
    )
    parser.add_argument(
        "--save-weight-dir",
        type=str,
        default=None,
        help="Override cfg.train.save_weight_dir. Each run should typically "
             "have its own subdirectory (e.g., checkpoints/.../run_1).",
    )
    args = parser.parse_args()

    # 1) Load config
    cfg = OmegaConf.load(args.config)

    if args.save_weight_dir:
        print(
            f"Overriding cfg.train.save_weight_dir: "
            f"'{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'"
        )
        cfg.train.save_weight_dir = args.save_weight_dir

    save_dir = cfg.train.save_weight_dir
    os.makedirs(save_dir, exist_ok=True)

    # 2) Load training AnnData
    print(f"Loading training AnnData from: {os.path.abspath(args.train_data_path)}")
    adata = sc.read_h5ad(args.train_data_path)

    latent = adata.obsm.get(args.latent_key, None)
    if latent is None:
        raise KeyError(
            f"adata.obsm['{args.latent_key}'] not found. "
            f"Please run your encoder and store its latent under this key."
        )

    n_latent = latent.shape[1]
    print(f"[Info] Detected latent dim from '{args.latent_key}': {n_latent}")
    if getattr(cfg.model.ae, "latent_dim", None) is None or cfg.model.ae.latent_dim != n_latent:
        print(
            f"Overriding cfg.model.ae.latent_dim: "
            f"'{cfg.model.ae.latent_dim}' -> '{n_latent}'"
        )
        cfg.model.ae.latent_dim = n_latent

    # gene dimension should match adata.n_vars
    if getattr(cfg.model.ae, "input_dim", None) != adata.n_vars:
        print(
            f"Overriding cfg.model.ae.input_dim: "
            f"'{cfg.model.ae.input_dim}' -> '{adata.n_vars}'"
        )
        cfg.model.ae.input_dim = adata.n_vars

    # 3) Prepare dataset / dataloader
    dataset = PairedLatentGeneDataset(adata, latent_key=args.latent_key)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    # 4) Build model
    device = torch.device(cfg.train.device)
    model = ScviLatentDDPMMLP(cfg).to(device)

    # 5) Optimizer & schedulers
    optim = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0.0)
    sched = GradualWarmupScheduler(
        optim,
        multiplier=cfg.train.warmup_multiplier,
        warm_epoch=cfg.train.epoch // 10,
        after_scheduler=cosine,
    )

    # 6) Resume from checkpoint if available
    latest_ckpt_path, is_final = find_latest_checkpoint(save_dir)
    start_epoch = 0
    global_step = 0

    if latest_ckpt_path is not None:
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)

        if is_final:
            print(
                f"[Resume] Found final checkpoint '{latest_ckpt_path}'. "
                f"Training is considered complete. Skipping."
            )
            return
        else:
            print(
                f"[Resume] Resuming from checkpoint '{latest_ckpt_path}' "
                f"starting at epoch {start_epoch+1}."
            )
    else:
        print("[Resume] No checkpoint found. Starting training from scratch.")

    # 7) Training loop
    print("Start training ScviLatentDDPMMLP (generic latent encoder)...")
    model.train()

    for epoch in range(start_epoch, cfg.train.epoch):
        for z0, z1, x1 in loader:
            z0 = z0.to(device)
            z1 = z1.to(device)
            x1 = x1.to(device)

            optim.zero_grad()
            loss_total, loss_diff, loss_dec = model.compute_loss(z0, z1, x1)
            loss_total.backward()
            optim.step()

            global_step += 1

        sched.step()

        print(
            f"Epoch {epoch+1}/{cfg.train.epoch} "
            f"loss_total={loss_total.item():.4f} "
            f"loss_diff={loss_diff.item():.4f} "
            f"loss_dec={loss_dec.item():.4f} "
            f"lr={sched.get_last_lr()[0]:.6f}"
        )

        # save intermediate checkpoint
        if (epoch + 1) % cfg.train.ckpt_save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved at: {ckpt_path}")

    # 8) Save final model
    final_path = os.path.join(save_dir, "model_final.pth")
    torch.save(
        {
            "epoch": cfg.train.epoch,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
        },
        final_path,
    )
    print(f"Final model saved at: {final_path}")


if __name__ == "__main__":
    main()
