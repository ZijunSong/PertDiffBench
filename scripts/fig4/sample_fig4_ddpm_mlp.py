#!/usr/bin/env python3
"""
Fig4: 生成带 treatment_time 的 h5ad 供 eval_fig4_time_conditioned 使用。
当前 DDPM+MLP 无时间条件，使用 train 中 0h 细胞作为 control 采样，
将同一批生成结果分别标为 4h 和 6h 以跑通流程；后续可改为真实时间条件采样。
"""
import os
import sys
import argparse
import numpy as np
import scanpy as sc
from omegaconf import OmegaConf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines/mlp_ddpm_mlp.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train-h5ad", required=True, help="fig4_train.h5ad (must have treatment_time)")
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--gene-nums", type=int, default=3000)
    parser.add_argument("--time-key", default="treatment_time")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.model.ae.input_dim = args.gene_nums
    device = torch.device(cfg.train.device)
    model = MLPDDPMMLP(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
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
        pred = model.sample(X).cpu().numpy()

    # 同一批生成结果分别标为 4h 和 6h（各 n 个），便于 eval 按时间分组
    half = n // 2
    obs_4h = [{"treatment_time": "4h"} for _ in range(half)]
    obs_6h = [{"treatment_time": "6h"} for _ in range(n - half)]
    import pandas as pd
    obs = pd.DataFrame(obs_4h + obs_6h)
    pred_4h = pred[:half]
    pred_6h = pred[half:n]
    X_out = np.concatenate([pred_4h, pred_6h], axis=0)
    out = sc.AnnData(X_out, obs=obs, var=adata.var.copy())
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    out.write_h5ad(args.out_h5ad)
    print(f"Saved {out.n_obs} samples to {args.out_h5ad} (placeholder 4h/6h from 0h-conditioned sampling).")


if __name__ == "__main__":
    main()
