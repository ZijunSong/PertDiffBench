#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Behavior:
- If a checkpoint already exists in cfg.train.save_weight_dir, load it and exit without training.
- Otherwise, train as usual and save checkpoints + final weights.

Notes:
- We still read the AnnData to infer latent/input dims so the model can be constructed consistently.
- Checkpoint preference order: model_final.pth > latest model_epoch_*.pth
"""

import os
import re
import argparse
from collections import defaultdict
from glob import glob
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
import scanpy as sc

from src.diffusion_baselines.models.scimilarity_latent_ddpm_mlp import (
    ScimilarityLatentDDPMMLP,
)
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler


class PairedScimLatentGeneDataset(Dataset):
    """
    Dataset returns:
        z0: control latent (SCimilarity)
        z1: perturbed latent
        x1: perturbed gene expression

    Assumptions:
        - adata.X: gene expression
        - adata.obsm["X_scim"]: SCimilarity latent
        - adata.obs["perturbation_status"]: 'Control' or perturbation labels
    """

    def __init__(self, adata):
        super().__init__()
        self.adata = adata

        if "perturbation_status" not in adata.obs.columns:
            raise KeyError("adata.obs must contain 'perturbation_status'.")

        if "X_scim" not in adata.obsm:
            raise KeyError("adata.obsm['X_scim'] not found. Please run SCimilarity encoder first.")

        self.ctrl_mask = adata.obs["perturbation_status"] == "Control"
        self.ctrl_ids = np.where(self.ctrl_mask.values)[0]

        self.pert_indices = np.where(~self.ctrl_mask.values)[0]
        self.pert_labels = adata.obs["perturbation_status"].values[self.pert_indices]

        self.pert_groups = defaultdict(list)
        for idx, label in zip(self.pert_indices, self.pert_labels):
            self.pert_groups[label].append(idx)
        self.pert_labels_unique = list(self.pert_groups.keys())

        print(f"[Dataset] #control cells: {len(self.ctrl_ids)}")
        print(f"[Dataset] #perturbation groups: {len(self.pert_labels_unique)}")

    def __len__(self):
        return len(self.pert_indices)

    def __getitem__(self, idx):
        pert_cell_idx = self.pert_indices[idx]
        # Randomly pick a control cell as the "paired" control
        ctrl_idx = np.random.choice(self.ctrl_ids)

        Z = self.adata.obsm["X_scim"]
        X = self.adata.X

        z0 = Z[ctrl_idx]
        z1 = Z[pert_cell_idx]
        x1 = X[pert_cell_idx].toarray() if hasattr(X, "toarray") else X[pert_cell_idx]

        z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        z1 = torch.from_numpy(np.asarray(z1, dtype=np.float32))
        x1 = torch.from_numpy(np.asarray(x1, dtype=np.float32))

        return z0, z1, x1


def find_existing_checkpoint(save_dir: str) -> Optional[str]:
    """
    Return path to an existing checkpoint if found.
    Priority: model_final.pth > latest model_epoch_*.pth
    """
    final_path = os.path.join(save_dir, "model_final.pth")
    if os.path.isfile(final_path):
        return final_path

    epoch_ckpts = glob(os.path.join(save_dir, "model_epoch_*.pth"))
    if not epoch_ckpts:
        return None

    # Extract epoch numbers and pick the largest
    def _epoch_num(p):
        m = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    epoch_ckpts.sort(key=_epoch_num, reverse=True)
    return epoch_ckpts[0]


def load_model_from_ckpt(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> Tuple[int, int]:
    """
    Load model/optimizer/scheduler states if available; return (epoch, step) from checkpoint.
    Optimizer/scheduler states are optional in "load-only" mode; here we only need the model.
    """
    print(f"[Load] Loading checkpoint from: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload)  # allow pure state_dict files

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys when loading: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[Warn] Unexpected keys when loading: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    epoch = int(payload.get("epoch", -1))
    step = int(payload.get("step", -1))
    return epoch, step


def maybe_override_dims_from_adata(cfg, adata):
    """
    Ensure cfg dimensions match data:
    - cfg.model.ae.latent_dim == adata.obsm['X_scim'].shape[1]
    - cfg.model.ae.input_dim == adata.n_vars
    """
    latent = adata.obsm.get("X_scim", None)
    if latent is None:
        raise KeyError("adata.obsm['X_scim'] not found. Run SCimilarity encoder first.")

    n_latent = latent.shape[1]
    if getattr(cfg.model.ae, "latent_dim", None) != n_latent:
        print(f"[Config] Overriding cfg.model.ae.latent_dim: '{cfg.model.ae.latent_dim}' -> '{n_latent}'")
        cfg.model.ae.latent_dim = int(n_latent)

    if getattr(cfg.model.ae, "input_dim", None) != adata.n_vars:
        print(f"[Config] Overriding cfg.model.ae.input_dim: '{cfg.model.ae.input_dim}' -> '{adata.n_vars}'")
        cfg.model.ae.input_dim = int(adata.n_vars)


def main():
    parser = argparse.ArgumentParser(
        description="Train DDPM-MLP in SCimilarity latent space, with decoder back to gene space. "
                    "If checkpoint exists, load and exit."
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to YAML config for ScimilarityLatentDDPMMLP."
    )
    parser.add_argument(
        "--train-data-path",
        required=True,
        help="AnnData (.h5ad) with obsm['X_scim'] (also used to set dims when loading).",
    )
    parser.add_argument(
        "--save-weight-dir",
        type=str,
        default=None,
        help="Override cfg.train.save_weight_dir.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Ignore existing checkpoints and train anyway.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.save_weight_dir:
        print(f"[Args] Override cfg.train.save_weight_dir: '{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'")
        cfg.train.save_weight_dir = args.save_weight_dir

    save_dir = cfg.train.save_weight_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Data] Loading AnnData: {os.path.abspath(args.train_data_path)}")
    adata = sc.read_h5ad(args.train_data_path)
    maybe_override_dims_from_adata(cfg, adata)

    device = torch.device(cfg.train.device)
    model = ScimilarityLatentDDPMMLP(cfg).to(device)

    # ---- Load-if-exists branch ----
    ckpt_path = find_existing_checkpoint(save_dir)
    if ckpt_path and not args.force_train:
        print("[Mode] Found existing checkpoint. Loading model and exiting without training.")
        _epoch, _step = load_model_from_ckpt(model, ckpt_path, device)
        model.eval()
        # Optional quick dry-run to ensure compatibility
        with torch.no_grad():
            # Create a tiny dummy batch from data to verify shapes
            Z = adata.obsm["X_scim"]
            X = adata.X
            any_idx = 0
            z0 = torch.from_numpy(np.asarray(Z[any_idx], dtype=np.float32)).unsqueeze(0).to(device)
            z1 = torch.from_numpy(np.asarray(Z[any_idx], dtype=np.float32)).unsqueeze(0).to(device)
            x1_np = X[any_idx].toarray() if hasattr(X, "toarray") else X[any_idx]
            x1 = torch.from_numpy(np.asarray(x1_np, dtype=np.float32)).unsqueeze(0).to(device)
            # Forward loss just to sanity check; ignore value
            _ = model.compute_loss(z0, z1, x1)
        print(f"[Load] Success. epoch={_epoch}, step={_step}. Exiting.")
        return

    # ---- Training branch ----
    print("[Mode] No usable checkpoint found (or --force-train set). Start training.")
    dataset = PairedScimLatentGeneDataset(adata)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    optim = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched = GradualWarmupScheduler(
        optim,
        multiplier=cfg.train.warmup_multiplier,
        warm_epoch=max(1, cfg.train.epoch // 10),
        after_scheduler=cosine,
    )

    print("[Train] Start training ScimilarityLatentDDPMMLP...")
    model.train()
    global_step = 0

    for epoch in range(cfg.train.epoch):
        for z0, z1, x1 in loader:
            z0 = z0.to(device, non_blocking=True)
            z1 = z1.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            loss_total, loss_diff, loss_dec = model.compute_loss(z0, z1, x1)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg.train, "grad_clip_norm", 1.0))
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

        if (epoch + 1) % cfg.train.ckpt_save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                },
                ckpt_path,
            )
            print(f"[Train] Checkpoint saved at: {ckpt_path}")

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
    print(f"[Train] Final model saved at: {final_path}")


if __name__ == "__main__":
    main()
