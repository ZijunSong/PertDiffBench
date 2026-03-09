#!/usr/bin/env python3

import os
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


class PairedLatentGeneDatasetSCF(Dataset):
    """
    Dataset returns:
        z0: control latent (from scFoundation)
        z1: perturbed latent
        x1: perturbed gene expression

    Assumptions:
        - adata.X: gene expression (same as PertBench)
        - adata.obsm[latent_key]: scFoundation latent
        - adata.obs["perturbation_status"]: 'Control' or perturbation labels
    """
    def __init__(self, adata, latent_key="X_scfoundation"):
        super().__init__()
        self.adata = adata
        self.latent_key = latent_key

        if "perturbation_status" not in adata.obs.columns:
            raise KeyError("adata.obs must contain 'perturbation_status'.")

        if latent_key not in adata.obsm:
            raise KeyError(f"adata.obsm['{latent_key}'] not found. Attach scFoundation embeddings first.")

        self.ctrl_mask = adata.obs["perturbation_status"] == "Control"
        self.ctrl_ids = np.where(self.ctrl_mask.values)[0]

        self.pert_indices = np.where(~self.ctrl_mask.values)[0]
        self.pert_labels = adata.obs["perturbation_status"].values[self.pert_indices]

        self.pert_groups = defaultdict(list)
        for idx, label in zip(self.pert_indices, self.pert_labels):
            self.pert_groups[label].append(idx)
        self.pert_labels_unique = list(self.pert_groups.keys())

        print(f"[Dataset-SCF] #control cells: {len(self.ctrl_ids)}")
        print(f"[Dataset-SCF] #perturbation groups: {len(self.pert_labels_unique)}")

    def __len__(self):
        return len(self.pert_indices)

    def __getitem__(self, idx):
        pert_cell_idx = self.pert_indices[idx]
        pert_label = self.pert_labels[idx]

        ctrl_idx = np.random.choice(self.ctrl_ids)

        z_all = self.adata.obsm[self.latent_key]
        X = self.adata.X

        z0 = z_all[ctrl_idx]
        z1 = z_all[pert_cell_idx]
        x1 = X[pert_cell_idx].toarray() if hasattr(X, "toarray") else X[pert_cell_idx]

        z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        z1 = torch.from_numpy(np.asarray(z1, dtype=np.float32))
        x1 = torch.from_numpy(np.asarray(x1, dtype=np.float32))

        return z0, z1, x1


def main():
    parser = argparse.ArgumentParser(
        description="Train DDPM-MLP in scFoundation latent space, with decoder back to gene space."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config for ScviLatentDDPMMLP (same as scVI version).",
    )
    parser.add_argument(
        "--train-data-path",
        required=True,
        help="Path to training AnnData (.h5ad) with obsm['X_scfoundation'].",
    )
    parser.add_argument(
        "--save-weight-dir",
        type=str,
        default=None,
        help="Override cfg.train.save_weight_dir.",
    )
    parser.add_argument(
        "--latent-key",
        type=str,
        default="X_scfoundation",
        help="obsm key for scFoundation latent. Default: X_scfoundation.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.save_weight_dir:
        print(
            f"Overriding cfg.train.save_weight_dir: '{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'"
        )
        cfg.train.save_weight_dir = args.save_weight_dir

    save_dir = cfg.train.save_weight_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading training AnnData from: {os.path.abspath(args.train_data_path)}")
    adata = sc.read_h5ad(args.train_data_path)

    latent = adata.obsm.get(args.latent_key, None)
    if latent is None:
        raise KeyError(f"adata.obsm['{args.latent_key}'] not found. Attach scFoundation embeddings first.")

    n_latent = latent.shape[1]
    print(f"[Info] Detected scFoundation latent dim: {n_latent}")
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

    dataset = PairedLatentGeneDatasetSCF(adata, latent_key=args.latent_key)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    device = torch.device(cfg.train.device)
    model = ScviLatentDDPMMLP(cfg).to(device)

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

    print("Start training ScFoundation-latent DDPM+decoder...")
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
