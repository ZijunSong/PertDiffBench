#!/usr/bin/env python3
"""
Fig4: Squidiff time imputation following the original paper / sample_squidiff.py.

Official Squidiff latent manipulation (bioRxiv Fig. 1E–G):
  - addition (default): z_interp = mean(z_origin) + Δz_sem * scale + noise,
    then DDIM decode conditioned on z_mod at every denoising step.
  - lerp: per-cell linear interpolation z = (1-α)*z_start + α*z_end, then DDIM decode.

For fig4 (anchor 2h→8h, impute 4h/6h):
  scale(4h) = (4-2)/(8-2) = 1/3
  scale(6h) = (6-2)/(8-2) = 2/3
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _project_root)
_src_squidiff = os.path.join(_project_root, "src", "Squidiff")
if _src_squidiff not in sys.path:
    sys.path.insert(0, _src_squidiff)

from Squidiff import dist_util


def _load_sampler_class():
    sample_path = os.path.join(_project_root, "src", "Squidiff", "sample_squidiff.py")
    spec = importlib.util.spec_from_file_location("sample_squidiff_fig4", sample_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sampler


def _toarray(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _time_to_hours(label: str) -> float:
    s = str(label).strip().lower().rstrip("h")
    return float(s)


def _scale_between(target_h: float, start_h: float, end_h: float) -> float:
    if end_h == start_h:
        raise ValueError("anchor start and end times must differ")
    return (target_h - start_h) / (end_h - start_h)


def _decode_with_zmod(sampler_obj, z_mod, gene_size, batch_size):
    sample_fn = sampler_obj.sample_fn
    model = sampler_obj.model
    device = dist_util.dev()
    out_list = []
    for start in range(0, z_mod.shape[0], batch_size):
        end = min(start + batch_size, z_mod.shape[0])
        z = z_mod[start:end]
        with torch.no_grad():
            sample = sample_fn(
                model,
                shape=(z.shape[0], gene_size),
                model_kwargs={"z_mod": z},
                device=device,
                noise=None,
            )
        out_list.append(sample.cpu().numpy())
    return np.concatenate(out_list, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Fig4 Squidiff time imputation (official addition / lerp + DDIM decode)"
    )
    parser.add_argument("--model_path", required=True, help="Squidiff model.pt")
    parser.add_argument("--train-h5ad", required=True, help="fig4_train.h5ad")
    parser.add_argument("--out-h5ad", required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--gene-size", type=int, default=3000)
    parser.add_argument("--output-dim", type=int, default=3000)
    parser.add_argument("--time-key", default="treatment_time")
    parser.add_argument("--anchor-start", default="2h", help="Early anchor time label")
    parser.add_argument("--anchor-end", default="8h", help="Late anchor time label")
    parser.add_argument(
        "--target-times",
        nargs="+",
        default=["4h", "6h"],
        help="Time labels to impute (scales computed from anchor range)",
    )
    parser.add_argument(
        "--method",
        choices=["addition", "lerp"],
        default="addition",
        help="addition: official interp_with_direction; lerp: per-cell linear interpolation",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable cell-wise noise around interpolated latent (addition only)",
    )
    args = parser.parse_args()

    from utils.seed import resolve_seed, set_seed
    _seed = resolve_seed(args.seed)
    set_seed(_seed)

    sampler_cls = _load_sampler_class()
    my_sampler = sampler_cls(
        model_path=args.model_path,
        gene_size=args.gene_size,
        output_dim=args.output_dim,
        use_drug_structure=False,
    )
    device = dist_util.dev()
    model = my_sampler.model

    adata = sc.read_h5ad(args.train_h5ad)
    if args.time_key not in adata.obs.columns:
        raise KeyError(f"Train adata must have obs['{args.time_key}']")
    time_vals = adata.obs[args.time_key].astype(str).str.strip()

    mask_start = time_vals == args.anchor_start
    mask_end = time_vals == args.anchor_end
    if not mask_start.any() or not mask_end.any():
        raise ValueError(
            f"Need both {args.anchor_start} and {args.anchor_end} cells in train."
        )

    adata_start = adata[mask_start]
    adata_end = adata[mask_end]
    from utils.max_eval_samples import resolve_eval_n_samples
    if args.n_samples is None or args.n_samples <= 0:
        args.n_samples = resolve_eval_n_samples(args.test_h5ad, 0, mode="timepoint")
        print(f"Using n_samples={args.n_samples}")

    n = min(args.n_samples, adata_start.n_obs, adata_end.n_obs)
    if n < 1:
        raise ValueError("Need at least one cell at each anchor time.")

    idx_start = np.random.choice(adata_start.n_obs, n, replace=(n > adata_start.n_obs))
    idx_end = np.random.choice(adata_end.n_obs, n, replace=(n > adata_end.n_obs))

    X_start = torch.tensor(
        _toarray(adata_start[idx_start].X).astype(np.float32), device=device
    )
    X_end = torch.tensor(
        _toarray(adata_end[idx_end].X).astype(np.float32), device=device
    )

    with torch.no_grad():
        z_start = model.encoder(X_start)
        z_end = model.encoder(X_end)

    start_h = _time_to_hours(args.anchor_start)
    end_h = _time_to_hours(args.anchor_end)
    direction = z_end.mean(dim=0) - z_start.mean(dim=0)

    all_X = []
    all_obs = []
    for target in args.target_times:
        alpha = _scale_between(_time_to_hours(target), start_h, end_h)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        print(
            f"[{args.method}] impute {target}: alpha={alpha:.4f} "
            f"(between {args.anchor_start} and {args.anchor_end})"
        )

        if args.method == "addition":
            pred = my_sampler.interp_with_direction(
                z_sem_origin=z_start,
                gene_size=args.gene_size,
                direction=direction,
                scale=alpha,
                add_noise_term=not args.no_noise,
            )
            x_pred = pred.detach().cpu().numpy()
        else:
            z_interp = (1.0 - alpha) * z_start + alpha * z_end
            x_pred = _decode_with_zmod(
                my_sampler, z_interp, args.gene_size, args.batch_size
            )

        x_pred = np.clip(x_pred, -1.0, 1.0)
        all_X.append(x_pred)
        all_obs.append(pd.DataFrame({args.time_key: [target] * n}))

    X_out = np.concatenate(all_X, axis=0)
    obs = pd.concat(all_obs, ignore_index=True)
    out = sc.AnnData(X_out, obs=obs, var=adata.var.copy())
    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    out.write_h5ad(args.out_h5ad)
    print(
        f"Saved {out.n_obs} samples to {args.out_h5ad} "
        f"(Squidiff {args.method}: {', '.join(args.target_times)})."
    )


if __name__ == "__main__":
    main()
