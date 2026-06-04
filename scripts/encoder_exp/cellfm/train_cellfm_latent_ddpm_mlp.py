# scripts/encoder_exp/cellfm/train_cellfm_latent_ddpm_mlp.py
#!/usr/bin/env python3

import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf

from src.diffusion_baselines.models.scvi_latent_ddpm_mlp import ScviLatentDDPMMLP
from src.diffusion_baselines.schedulers.warmup import GradualWarmupScheduler


class PairedCellfmLatentGeneDataset(Dataset):
    """
    Dataset returns:
        z0: control latent (X_cellfm)
        z1: perturbed latent (X_cellfm)
        x1: perturbed gene expression (adata.X)

    Assume: 
      - adata.X: gene expression matrix
      - adata.obsm["X_cellfm"]: latent from CellFM encoder
      - adata.obs["perturbation_status"]: 'Control' or perturbation labels
    """
    def __init__(self, adata):
        super().__init__()
        self.adata = adata

        if "perturbation_status" not in adata.obs.columns:
            raise KeyError("adata.obs must contain 'perturbation_status'.")
        if "X_cellfm" not in adata.obsm:
            raise KeyError("adata.obsm['X_cellfm'] not found. Run CellFM encoder first.")

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
        pert_label = self.pert_labels[idx]

        ctrl_idx = np.random.choice(self.ctrl_ids)

        z_all = self.adata.obsm["X_cellfm"]
        X = self.adata.X

        z0 = z_all[ctrl_idx]
        z1 = z_all[pert_cell_idx]
        x1 = X[pert_cell_idx].toarray() if hasattr(X, "toarray") else X[pert_cell_idx]

        z0 = torch.from_numpy(np.asarray(z0, dtype=np.float32))
        z1 = torch.from_numpy(np.asarray(z1, dtype=np.float32))
        x1 = torch.from_numpy(np.asarray(x1, dtype=np.float32))

        return z0, z1, x1


def find_last_epoch_ckpt(save_dir: str):
    pattern = os.path.join(save_dir, "model_epoch_*.pth")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None, 0
    def _epoch_from_name(path):
        base = os.path.basename(path)
        num = base.replace("model_epoch_", "").replace(".pth", "")
        try:
            return int(num)
        except ValueError:
            return -1
    ckpts = [(path, _epoch_from_name(path)) for path in ckpts]
    ckpts = [p for p in ckpts if p[1] > 0]
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: x[1])
    last_path, last_epoch = ckpts[-1]
    return last_path, last_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Train DDPM-MLP in CellFM latent space, with decoder to gene space (resume supported)."
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config for ScviLatentDDPMMLP (same structure as scvi_ddpm_mlp.yaml).",
    )
    parser.add_argument(
        "--train-data-path",
        required=True,
        help="Training AnnData (.h5ad) with obsm['X_cellfm'] and obs['perturbation_status'].",
    )
    parser.add_argument(
        "--save-weight-dir",
        type=str,
        default=None,
        help="Override cfg.train.save_weight_dir.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.save_weight_dir:
        print(
            f"[Config] Override cfg.train.save_weight_dir: "
            f"'{cfg.train.save_weight_dir}' -> '{args.save_weight_dir}'"
        )
        cfg.train.save_weight_dir = args.save_weight_dir

    save_dir = cfg.train.save_weight_dir
    os.makedirs(save_dir, exist_ok=True)

    final_ckpt = os.path.join(save_dir, "model_final.pth")
    if os.path.exists(final_ckpt):
        print(f"[Train] Found final checkpoint at {final_ckpt}, skip training.")
        return

    print(f"[Train] Loading training AnnData from: {os.path.abspath(args.train_data_path)}")
    adata = sc.read_h5ad(args.train_data_path)

    latent = adata.obsm.get("X_cellfm", None)
    if latent is None:
        raise KeyError("adata.obsm['X_cellfm'] not found. Run CellFM encoder first.")

    n_latent = latent.shape[1]
    print(f"[Info] Detected CellFM latent dim: {n_latent}")

    if getattr(cfg.model.ae, "latent_dim", None) != n_latent:
        print(
            f"[Config] Override cfg.model.ae.latent_dim: "
            f"{cfg.model.ae.latent_dim} -> {n_latent}"
        )
        cfg.model.ae.latent_dim = n_latent

    if getattr(cfg.model.ae, "input_dim", None) != adata.n_vars:
        print(
            f"[Config] Override cfg.model.ae.input_dim: "
            f"{cfg.model.ae.input_dim} -> {adata.n_vars}"
        )
        cfg.model.ae.input_dim = adata.n_vars

    dataset = PairedCellfmLatentGeneDataset(adata)
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

    start_epoch = 0
    global_step = 0

    last_ckpt, last_epoch = find_last_epoch_ckpt(save_dir)
    if last_ckpt is not None:
        print(f"[Resume] Found checkpoint {last_ckpt} (epoch {last_epoch}), loading...")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", last_epoch) + 1
        global_step = ckpt.get("step", 0)
        print(f"[Resume] Resume from epoch {start_epoch}, global_step {global_step}.")
    else:
        print("[Train] No previous checkpoint, start from scratch.")

    print("Start training ScviLatentDDPMMLP (CellFM latent)...")
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

    torch.save(
        {
            "epoch": cfg.train.epoch,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
        },
        final_ckpt,
    )
    print(f"Final model saved at: {final_ckpt}")


if __name__ == "__main__":
    main()
