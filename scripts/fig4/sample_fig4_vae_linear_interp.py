#!/usr/bin/env python3
"""
Fig4: using VAE (encoder → 2h/8h latent linear interpolation → decoder) 4h/6h cell.
 DDPM and DDPM+MLP baseline using: diffusion, onlyusing encoder/decoder linear interpolation.
--ckpt canas: (1) DDPM+MLP model_epoch_1000.pth ( ); (2) DDPM using ae_epoch_1000.pth (only encoder/decoder, to strict=False ).
"""
import os
import sys
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from omegaconf import OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP


def main():
    parser = argparse.ArgumentParser(description="Fig4: VAE encoder → linear interp (2h/8h) → decoder → 4h/6h")
    parser.add_argument("--config", default="configs/baselines/mlp_ddpm_mlp.yaml")
    parser.add_argument("--ckpt", required=True, help="MLPDDPMMLP checkpoint (has encoder/decoder)")
    parser.add_argument("--train-h5ad", required=True, help="fig4_train.h5ad (must have treatment_time)")
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--n-samples", type=int, default=500, help="number of cells per time (4h and 6h)")
    parser.add_argument("--gene-nums", type=int, default=3000)
    parser.add_argument("--time-key", default="treatment_time")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    cfg.model.ae.input_dim = args.gene_nums
    device = torch.device(cfg.train.device)
    model = MLPDDPMMLP(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # supportonlywith encoder/decoder checkpoint ( DDPM using AE), using strict=False
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Info] Loaded AE-only ckpt: {len(missing)} keys not in model (diffusion part).")
    model.eval()

    adata = sc.read_h5ad(args.train_h5ad)
    if args.time_key not in adata.obs.columns:
        raise KeyError(f"Train adata must have obs['{args.time_key}']")
    time_vals = adata.obs[args.time_key].astype(str).str.strip()

    mask_2h = time_vals == "2h"
    mask_8h = time_vals == "8h"
    if not mask_2h.any():
        raise ValueError("No 2h cells in train.")
    if not mask_8h.any():
        raise ValueError("No 8h cells in train.")

    adata_2h = adata[mask_2h]
    adata_8h = adata[mask_8h]
    n_2h, n_8h = adata_2h.n_obs, adata_8h.n_obs
    n = min(args.n_samples, n_2h, n_8h)
    if n < 1:
        raise ValueError("Need at least one 2h and one 8h cell.")

    idx_2h = np.random.choice(n_2h, n, replace=(n > n_2h))
    idx_8h = np.random.choice(n_8h, n, replace=(n > n_8h))
    X_2h = adata_2h[idx_2h].X
    X_8h = adata_8h[idx_8h].X
    X_2h = X_2h.toarray() if hasattr(X_2h, "toarray") else np.asarray(X_2h)
    X_8h = X_8h.toarray() if hasattr(X_8h, "toarray") else np.asarray(X_8h)
    X_2h = torch.from_numpy(X_2h.astype(np.float32)).to(device)
    X_8h = torch.from_numpy(X_8h.astype(np.float32)).to(device)

    with torch.no_grad():
        z_2h = model.encoder(X_2h)
        z_8h = model.encoder(X_8h)
        # 4h = 0.5*2h + 0.5*8h, 6h = 0.25*2h + 0.75*8h
        z_4h = 0.5 * z_2h + 0.5 * z_8h
        z_6h = 0.25 * z_2h + 0.75 * z_8h
        x_4h = model.decoder(z_4h)
        x_6h = model.decoder(z_6h)
        x_4h = x_4h.clamp(-1, 1).cpu().numpy()
        x_6h = x_6h.clamp(-1, 1).cpu().numpy()

    obs_4h = pd.DataFrame({"treatment_time": ["4h"] * n})
    obs_6h = pd.DataFrame({"treatment_time": ["6h"] * n})
    obs = pd.concat([obs_4h, obs_6h], ignore_index=True)
    X_out = np.concatenate([x_4h, x_6h], axis=0)
    out = sc.AnnData(X_out, obs=obs, var=adata.var.copy())
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    out.write_h5ad(args.out_h5ad)
    print(f"Saved {out.n_obs} samples to {args.out_h5ad} (VAE linear interp: 4h={n}, 6h={n}).")


if __name__ == "__main__":
    main()
