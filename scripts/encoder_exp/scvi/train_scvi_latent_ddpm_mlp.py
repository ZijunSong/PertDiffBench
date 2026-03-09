#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
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
    Dataset 返回:
        z0: control latent
        z1: perturbed latent
        x1: perturbed gene expression

    假设:
        - adata.X: gene expression
        - adata.obsm["X_scvi"]: latent
        - adata.obs["perturbation_status"]: 'Control' or perturbation labels
    """
    def __init__(self, adata):
        super().__init__()
        self.adata = adata

        if "perturbation_status" not in adata.obs.columns:
            raise KeyError("adata.obs must contain 'perturbation_status'.")

        if "X_scvi" not in adata.obsm:
            raise KeyError("adata.obsm['X_scvi'] not found. Please run scVI encoder first.")

        self.ctrl_mask = adata.obs["perturbation_status"] == "Control"
        self.ctrl_ids = np.where(self.ctrl_mask.values)[0]

        self.pert_indices = np.where(~self.ctrl_mask.values)[0]
        self.pert_labels = adata.obs["perturbation_status"].values[self.pert_indices]

        # group by perturbation for sampling convenience
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
        pert_label = self.pert_labels[idx]

        ctrl_idx = np.random.choice(self.ctrl_ids)

        z_all = self.adata.obsm["X_scvi"]
        X = self.adata.X

        z0 = z_all[ctrl_idx]
        z1 = z_all[pert_cell_idx]
        x1 = X[pert_cell_idx].toarray() if hasattr(X, "toarray") else X[pert_cell_idx]

        z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        z1 = torch.from_numpy(np.asarray(z1, dtype=np.float32))
        x1 = torch.from_numpy(np.asarray(x1, dtype=np.float32))

        return z0, z1, x1


def try_find_checkpoint(save_dir: str):
    """
    优先返回:
      1) model_final.pth
      2) 最新的 model_epoch_*.pth
    """
    final_path = os.path.join(save_dir, "model_final.pth")
    if os.path.isfile(final_path):
        return final_path

    # find latest epoch checkpoint
    pattern = re.compile(r"model_epoch_(\d+)\.pth$")
    latest = (-1, None)
    if os.path.isdir(save_dir):
        for fname in os.listdir(save_dir):
            m = pattern.match(fname)
            if m:
                ep = int(m.group(1))
                if ep > latest[0]:
                    latest = (ep, os.path.join(save_dir, fname))
    return latest[1]


def load_model_weights(model, ckpt_path: str, device):
    """
    仅加载模型权重；你也可以按需加载优化器/调度器状态以便 resume 训练。
    这里遵循“已训就用，不再训练”的语义，只加载 model。
    """
    print(f"[Info] Loading checkpoint: {os.path.abspath(ckpt_path)}")
    state = torch.load(ckpt_path, map_location=device)

    # 常见字段名
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        # 兜底：如果直接存的 state_dict
        model.load_state_dict(state, strict=True)

    print(f"[OK] Loaded pre-trained model from: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train DDPM-MLP in scVI latent space, with decoder back to gene space."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config for ScviLatentDDPMMLP (similar to mlp_ddpm_mlp, but using latent_dim).",
    )
    parser.add_argument(
        "--train-data-path",
        required=True,
        help="Path to training AnnData (.h5ad) that already has obsm['X_scvi'].",
    )
    parser.add_argument(
        "--save-weight-dir",
        type=str,
        default=None,
        help="Override cfg.train.save_weight_dir.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoints and train from scratch.",
    )
    args = parser.parse_args()

    # 1) Load config
    cfg = OmegaConf.load(args.config)

    if args.save_weight_dir:
        print(
            f"Overriding cfg.train.save_weight_dir: '{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'"
        )
        cfg.train.save_weight_dir = args.save_weight_dir

    save_dir = cfg.train.save_weight_dir
    os.makedirs(save_dir, exist_ok=True)

    # 2) Load training AnnData (with X_scvi)
    print(f"Loading training AnnData from: {os.path.abspath(args.train_data_path)}")
    adata = sc.read_h5ad(args.train_data_path)

    latent = adata.obsm.get("X_scvi", None)
    if latent is None:
        raise KeyError("adata.obsm['X_scvi'] not found. Run scVI encoder first.")

    # Infer dims from data to keep model/config consistent before loading weights
    n_latent = latent.shape[1]
    print(f"[Info] Detected scVI latent dim: {n_latent}")
    if getattr(cfg.model.ae, "latent_dim", None) is None or cfg.model.ae.latent_dim != n_latent:
        print(
            f"Overriding cfg.model.ae.latent_dim: '{cfg.model.ae.latent_dim}' -> '{n_latent}'"
        )
        cfg.model.ae.latent_dim = n_latent

    if getattr(cfg.model.ae, "input_dim", None) != adata.n_vars:
        print(
            f"Overriding cfg.model.ae.input_dim: '{cfg.model.ae.input_dim}' -> '{adata.n_vars}'"
        )
        cfg.model.ae.input_dim = adata.n_vars

    # 3) Prepare dataset / dataloader
    dataset = PairedLatentGeneDataset(adata)
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

    # 4.1) Auto-load pre-trained model if exists (and not forcing retrain)
    ckpt_path = try_find_checkpoint(save_dir)
    if ckpt_path and not args.force_retrain:
        load_model_weights(model, ckpt_path, device)
        print("[Exit] Pre-trained model found and loaded. Skip training to avoid retraining.")
        return

    # 5) Optimizer & schedulers (only set up when actually training)
    optim = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    cosine = CosineAnnealingLR(optim, T_max=cfg.train.epoch, eta_min=0)
    sched = GradualWarmupScheduler(
        optim,
        multiplier=cfg.train.warmup_multiplier,
        warm_epoch=cfg.train.epoch // 10,
        after_scheduler=cosine,
    )

    # 6) Training loop
    print("Start training ScviLatentDDPMMLP...")
    model.train()
    global_step = 0

    for epoch in range(cfg.train.epoch):
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

        # Save checkpoint
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
            print(f"Checkpoint saved at: {ckpt_path}")

    # Save final
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
