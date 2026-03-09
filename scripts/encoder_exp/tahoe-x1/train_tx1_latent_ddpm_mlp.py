#!/usr/bin/env python3
from __future__ import annotations

"""
Train DDPM in a *precomputed* latent space (e.g., Tahoe-x1 / Tx1 embeddings) with a gene-space decoder.

This is intentionally parallel to your existing scVI pipeline:
  raw scRNA -> (encoder) -> latent z -> DDPM(z0->z1_hat) + decoder -> x1_hat

Differences vs scVI version:
- encoder is *external* (Tx1), so this script only consumes an h5ad that already has embeddings in adata.obsm[--obsm-key]
- robust resume:
    * if model_final.pth exists, we skip by default
    * else auto-resume from the latest model_epoch_*.pth in save_weight_dir

Example:
  python scripts/encoder_exp/train_tx1_latent_ddpm_mlp.py \
      -c configs/baselines/tx1_ddpm_mlp.yaml \
      --train-data-path samples/encoder_exp/tx1_ddpm/task1_train_CD4T_with_tx1_latent.h5ad \
      --obsm-key X_tx1 \
      --save-weight-dir checkpoints/tx1_ddpm/latent_ddpm \
      --resume auto
"""

import os
import sys
# Ensure repo root is on PYTHONPATH so `src.*` and `utils.*` imports work even when called from subdirs.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import os
import re
import glob
import argparse
from collections import defaultdict
from typing import Optional, Tuple

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
    Returns:
        z0: control latent
        z1: perturbed latent
        x1: perturbed gene expression

    Assumptions:
        - adata.X: gene expression (dense or sparse)
        - adata.obsm[obsm_key]: latent embeddings
        - adata.obs["perturbation_status"]: 'Control' or perturbation labels

    NOTE: Pairing strategy here is intentionally simple (random control for each perturbed cell).
    For tighter biological pairing (e.g., matched batch / donor), refine this sampler.
    """
    def __init__(self, adata, obsm_key: str):
        super().__init__()
        self.adata = adata
        self.obsm_key = obsm_key

        if "perturbation_status" not in adata.obs.columns:
            raise KeyError("adata.obs must contain 'perturbation_status'.")

        if obsm_key not in adata.obsm:
            raise KeyError(f"adata.obsm['{obsm_key}'] not found. Run the encoder first.")

        self.ctrl_mask = adata.obs["perturbation_status"] == "Control"
        self.ctrl_ids = np.where(self.ctrl_mask.values)[0]

        self.pert_indices = np.where(~self.ctrl_mask.values)[0]
        self.pert_labels = adata.obs["perturbation_status"].values[self.pert_indices]

        self.pert_groups = defaultdict(list)
        for idx, label in zip(self.pert_indices, self.pert_labels):
            self.pert_groups[label].append(idx)
        self.pert_labels_unique = list(self.pert_groups.keys())

        print(f"[Dataset] obsm_key={obsm_key}")
        print(f"[Dataset] #control cells: {len(self.ctrl_ids)}")
        print(f"[Dataset] #perturbation groups: {len(self.pert_labels_unique)}")
        print(f"[Dataset] #perturbed cells: {len(self.pert_indices)}")

    def __len__(self):
        return len(self.pert_indices)

    def __getitem__(self, idx):
        pert_cell_idx = self.pert_indices[idx]

        ctrl_idx = np.random.choice(self.ctrl_ids)

        z_all = self.adata.obsm[self.obsm_key]
        X = self.adata.X

        z0 = z_all[ctrl_idx]
        z1 = z_all[pert_cell_idx]
        x1 = X[pert_cell_idx].toarray() if hasattr(X, "toarray") else X[pert_cell_idx]

        z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        z1 = torch.from_numpy(np.asarray(z1, dtype=np.float32))
        x1 = torch.from_numpy(np.asarray(x1, dtype=np.float32))

        return z0, z1, x1


def _find_latest_ckpt(save_dir: str) -> Optional[str]:
    paths = glob.glob(os.path.join(save_dir, "model_epoch_*.pth"))
    if not paths:
        return None

    def _epoch(p: str) -> int:
        m = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths.sort(key=_epoch)
    return paths[-1]


def _load_ckpt(
    ckpt_path: str,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    sched,
    device: torch.device,
) -> Tuple[int, int]:
    print(f"[Resume] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optim.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt and sched is not None:
        try:
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[Resume] Warning: failed to load scheduler_state_dict: {e}")

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("step", 0))
    print(f"[Resume] start_epoch={start_epoch}  global_step={global_step}")
    return start_epoch, global_step


def main():
    parser = argparse.ArgumentParser(description="Train DDPM+decoder using precomputed latent embeddings (Tx1/scVI/etc).")
    parser.add_argument("-c", "--config", required=True, help="YAML config for ScviLatentDDPMMLP.")
    parser.add_argument("--train-data-path", required=True, help="Training AnnData (.h5ad) with precomputed embeddings in .obsm.")
    parser.add_argument("--obsm-key", default="X_tx1", help="Which adata.obsm key holds the latent embeddings.")
    parser.add_argument("--save-weight-dir", type=str, default=None, help="Override cfg.train.save_weight_dir.")
    parser.add_argument("--resume", type=str, default="auto",
                        help="Resume mode: 'auto' (latest ckpt), 'none', or a specific checkpoint path.")
    parser.add_argument("--force", action="store_true", help="Re-train even if model_final.pth exists.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.save_weight_dir:
        print(f"Overriding cfg.train.save_weight_dir: '{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'")
        cfg.train.save_weight_dir = args.save_weight_dir

    save_dir = cfg.train.save_weight_dir
    os.makedirs(save_dir, exist_ok=True)

    final_path = os.path.join(save_dir, "model_final.pth")
    if os.path.exists(final_path) and (not args.force):
        print(f"[Skip] Found final checkpoint: {final_path} (use --force to retrain)")
        return

    print(f"Loading training AnnData from: {os.path.abspath(args.train_data_path)}")
    adata = sc.read_h5ad(args.train_data_path)

    latent = adata.obsm.get(args.obsm_key, None)
    if latent is None:
        raise KeyError(f"adata.obsm['{args.obsm_key}'] not found. Run encoder first.")

    n_latent = int(latent.shape[1])
    print(f"[Info] Detected latent dim from obsm['{args.obsm_key}']: {n_latent}")

    # Make config consistent with actual data
    if getattr(cfg.model.ae, "latent_dim", None) != n_latent:
        print(f"Overriding cfg.model.ae.latent_dim: '{cfg.model.ae.latent_dim}' -> '{n_latent}'")
        cfg.model.ae.latent_dim = n_latent

    if getattr(cfg.model.ae, "input_dim", None) != adata.n_vars:
        print(f"Overriding cfg.model.ae.input_dim: '{cfg.model.ae.input_dim}' -> '{adata.n_vars}'")
        cfg.model.ae.input_dim = int(adata.n_vars)

    dataset = PairedLatentGeneDataset(adata, obsm_key=args.obsm_key)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device(cfg.train.device)
    print(f"[Train] device={device}")

    model = ScviLatentDDPMMLP(cfg).to(device)

    optim = AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    cosine = CosineAnnealingLR(optim, T_max=int(cfg.train.epoch), eta_min=0)
    sched = GradualWarmupScheduler(
        optim,
        multiplier=float(cfg.train.warmup_multiplier),
        warm_epoch=int(cfg.train.epoch) // 10,
        after_scheduler=cosine,
    )

    # ---------- Resume ----------
    start_epoch = 0
    global_step = 0
    resume_mode = (args.resume or "auto").lower()
    ckpt_path = None
    if resume_mode == "none":
        ckpt_path = None
    elif resume_mode == "auto":
        ckpt_path = _find_latest_ckpt(save_dir)
    else:
        ckpt_path = args.resume

    if ckpt_path is not None and os.path.exists(ckpt_path):
        start_epoch, global_step = _load_ckpt(ckpt_path, model, optim, sched, device)
    elif ckpt_path is not None:
        print(f"[Resume] Requested ckpt not found: {ckpt_path}. Starting from scratch.")

    # ---------- Training loop ----------
    print(f"[Train] Start training for epoch={cfg.train.epoch} (start_epoch={start_epoch})")
    model.train()

    for epoch in range(start_epoch, int(cfg.train.epoch)):
        for z0, z1, x1 in loader:
            z0 = z0.to(device, non_blocking=True)
            z1 = z1.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            loss_total, loss_diff, loss_dec = model.compute_loss(z0, z1, x1)

            loss_total.backward()

            # Optional grad clip if present in cfg
            grad_clip = getattr(cfg.train, "grad_clip", None)
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

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

        # Save checkpoint
        ckpt_interval = OmegaConf.select(cfg, "train.ckpt_save_interval", default=50)
        if (epoch + 1) % int(ckpt_interval) == 0:
            ckpt_out = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_out,
            )
            print(f"Checkpoint saved: {ckpt_out}")

    # Final
    torch.save(
        {
            "epoch": int(cfg.train.epoch) - 1,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        },
        final_path,
    )
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()