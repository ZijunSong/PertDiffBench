#!/usr/bin/env python3
"""
Fig4: treatment_time h5ad eval_fig4_time_conditioned using.
current DDPM when rows , using train 0h cell as control , 
 results as 4h 6h to ; after can as when rows .
"""
import os
import sys
import argparse
import numpy as np
import scanpy as sc
from omegaconf import OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.diffusion_baselines.models.scrna_ddpm_scrna import ScrnaDDPM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines/scrna_ddpm_scrna.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train-h5ad", required=True)
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--gene-nums", type=int, default=3000)
    parser.add_argument("--time-key", default="treatment_time")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.input_dim = args.gene_nums
    device = torch.device(cfg.train.device)
    model = ScrnaDDPM(cfg).to(device)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    adata = sc.read_h5ad(args.train_h5ad)
    if args.time_key not in adata.obs.columns:
        raise KeyError(f"Train adata must have obs['{args.time_key}']")
    mask_0h = adata.obs[args.time_key].astype(str).str.strip() == "0h"
    if not mask_0h.any():
        raise ValueError("No 0h cells in train for control conditioning.")
    ctrl = adata[mask_0h]
    n = min(args.n_samples, ctrl.n_obs)
    np.random.seed(0)
    idx = np.random.choice(ctrl.n_obs, n, replace=(n > ctrl.n_obs))
    ctrl = ctrl[idx]
    X = ctrl.X.toarray() if hasattr(ctrl.X, "toarray") else np.asarray(ctrl.X)
    X = torch.from_numpy(X.astype(np.float32)).to(device)

    with torch.no_grad():
        pred = model.sample_cond(X).cpu().numpy()

    half = n // 2
    import pandas as pd
    obs = pd.DataFrame([{"treatment_time": "4h"} for _ in range(half)] + [{"treatment_time": "6h"} for _ in range(n - half)])
    X_out = np.concatenate([pred[:half], pred[half:n]], axis=0)
    out = sc.AnnData(X_out, obs=obs, var=adata.var.copy())
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    out.write_h5ad(args.out_h5ad)
    print(f"Saved {out.n_obs} samples to {args.out_h5ad} (placeholder 4h/6h from 0h-conditioned sampling).")


if __name__ == "__main__":
    main()
