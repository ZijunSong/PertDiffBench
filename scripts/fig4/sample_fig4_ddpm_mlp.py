#!/usr/bin/env python3
"""
Fig4: treatment_time h5ad eval_fig4_time_conditioned using.
current DDPM+MLP when rows , using train 0h cell as control , 
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
from src.diffusion_baselines.models.mlp_ddpm_mlp_diffusion import MLPDDPMMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baselines/mlp_ddpm_mlp.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train-h5ad", required=True, help="fig4_train.h5ad (must have treatment_time)")
    parser.add_argument("--test-h5ad", default="", help="fig4_test.h5ad for resolving max n_samples (recommended)")
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--n-samples", type=int, default=0, help="Cells to sample from 0h control (0 = max per time point from test h5ad)")
    parser.add_argument("--gene-nums", type=int, default=3000)
    parser.add_argument("--time-key", default="treatment_time")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (overridden by RUN_SEED env per run)")
    args = parser.parse_args()

    from utils.seed import resolve_seed, set_seed
    run_seed = resolve_seed(getattr(args, "seed", 0))
    set_seed(run_seed)

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
    from utils.max_eval_samples import resolve_eval_n_samples
    if args.n_samples is None or args.n_samples <= 0:
        eval_h5ad = args.test_h5ad or args.train_h5ad
        args.n_samples = resolve_eval_n_samples(eval_h5ad, 0, mode="timepoint", time_col=args.time_key)
        print(f"Using n_samples={args.n_samples}")

    n = min(args.n_samples, ctrl.n_obs)
    np.random.seed(run_seed)
    idx = np.random.choice(ctrl.n_obs, n, replace=(n > ctrl.n_obs))
    ctrl = ctrl[idx]
    X = ctrl.X.toarray() if hasattr(ctrl.X, "toarray") else np.asarray(ctrl.X)
    X = torch.from_numpy(X.astype(np.float32)).to(device)

    with torch.no_grad():
        pred = model.sample(X).cpu().numpy()

    # results as 4h 6h ( n ), eval when 
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
