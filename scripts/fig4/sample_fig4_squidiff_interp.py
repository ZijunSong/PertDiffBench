#!/usr/bin/env python3
"""
Fig4: Squidiff 在 latent 空间对 2h/8h 做线性插值得到 4h/6h latent，再经 diffusion 解码为表达，写入 h5ad 供 eval。
"""
import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
import scanpy as sc

# Project root and Squidiff path (script at scripts/fig4/ -> project root = 2 dirnames up)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _project_root)
# Squidiff 包在 src/Squidiff/Squidiff/，需把 src/Squidiff 加入 path
_src_squidiff = os.path.abspath(os.path.join(_project_root, "src", "Squidiff"))
if _src_squidiff not in sys.path:
    sys.path.insert(0, _src_squidiff)

from Squidiff import dist_util
from Squidiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


def main():
    parser = argparse.ArgumentParser(description="Fig4: Squidiff latent linear interp (2h/8h -> 4h/6h) then decode")
    parser.add_argument("--model_path", required=True, help="Squidiff model.pt (e.g. checkpoints/fig4/squidiff_3000/run1/model.pt)")
    parser.add_argument("--train-h5ad", required=True, help="fig4_train.h5ad")
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--gene-size", type=int, default=3000)
    parser.add_argument("--output-dim", type=int, default=3000)
    parser.add_argument("--time-key", default="treatment_time")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ddim", action="store_true", default=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Model/diffusion args (fig4: no drug)
    default_args = model_and_diffusion_defaults()
    default_args.update({
        "gene_size": args.gene_size,
        "output_dim": args.output_dim,
        "use_drug_structure": False,
        "class_cond": False,
        "use_encoder": True,
        "num_layers": 3,
        "diffusion_steps": 1000,
    })
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(default_args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path))
    device = dist_util.dev()
    model.to(device)
    model.eval()

    sample_fn = diffusion.ddim_sample_loop if default_args.get("use_ddim", True) else diffusion.p_sample_loop

    adata = sc.read_h5ad(args.train_h5ad)
    if args.time_key not in adata.obs.columns:
        raise KeyError(f"Train adata must have obs['{args.time_key}']")
    time_vals = adata.obs[args.time_key].astype(str).str.strip()
    mask_2h = time_vals == "2h"
    mask_8h = time_vals == "8h"
    if not mask_2h.any() or not mask_8h.any():
        raise ValueError("Need both 2h and 8h cells in train.")
    adata_2h = adata[mask_2h]
    adata_8h = adata[mask_8h]
    n_2h, n_8h = adata_2h.n_obs, adata_8h.n_obs
    n = min(args.n_samples, n_2h, n_8h)
    idx_2h = np.random.choice(n_2h, n, replace=(n > n_2h))
    idx_8h = np.random.choice(n_8h, n, replace=(n > n_8h))
    X_2h = adata_2h[idx_2h].X
    X_8h = adata_8h[idx_8h].X
    X_2h = X_2h.toarray() if hasattr(X_2h, "toarray") else np.asarray(X_2h)
    X_8h = X_8h.toarray() if hasattr(X_8h, "toarray") else np.asarray(X_8h)
    X_2h = torch.tensor(X_2h.astype(np.float32)).to(device)
    X_8h = torch.tensor(X_8h.astype(np.float32)).to(device)

    with torch.no_grad():
        z_2h = model.encoder(X_2h)
        z_8h = model.encoder(X_8h)
        z_4h = 0.5 * z_2h + 0.5 * z_8h
        z_6h = 0.25 * z_2h + 0.75 * z_8h

    # Decode via diffusion in batches
    def decode_batches(z_batch, batch_size=args.batch_size):
        out_list = []
        for start in range(0, z_batch.shape[0], batch_size):
            end = min(start + batch_size, z_batch.shape[0])
            z = z_batch[start:end]
            with torch.no_grad():
                sample = sample_fn(
                    model,
                    shape=(z.shape[0], args.gene_size),
                    model_kwargs={"z_mod": z},
                    device=device,
                )
            out_list.append(sample.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    x_4h = decode_batches(z_4h)
    x_6h = decode_batches(z_6h)
    x_4h = np.clip(x_4h, -1.0, 1.0)
    x_6h = np.clip(x_6h, -1.0, 1.0)

    obs_4h = pd.DataFrame({args.time_key: ["4h"] * n})
    obs_6h = pd.DataFrame({args.time_key: ["6h"] * n})
    obs = pd.concat([obs_4h, obs_6h], ignore_index=True)
    X_out = np.concatenate([x_4h, x_6h], axis=0)
    out = sc.AnnData(X_out, obs=obs, var=adata.var.copy())
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    out.write_h5ad(args.out_h5ad)
    print(f"Saved {out.n_obs} samples to {args.out_h5ad} (Squidiff latent linear interp: 4h={n}, 6h={n}).")


if __name__ == "__main__":
    main()
