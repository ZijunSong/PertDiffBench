#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig4 time-conditioned generation - usingeval .
input: test set h5ad (with treatment_time 4h/6h), results h5ad (with treatment_time), 
     optionaltrain set h5ad (for 0h as control delta ).
 treatment_time , foreachwhen 11 items , and / CSV.
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os

# repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.metrics import (
    compute_mae,
    compute_des,
    compute_pds,
    compute_edistance,
    compute_r2,
    compute_mmd,
    compute_pearson,
    compute_pearson_delta,
    compute_pearson_delta_de,
)


def _toarray(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def main():
    parser = argparse.ArgumentParser(description="Evaluate time-conditioned generation (fig4): per treatment_time metrics then average.")
    parser.add_argument("--test-h5ad", required=True, help="Test h5ad with obs.treatment_time (e.g. 4h, 6h)")
    parser.add_argument("--generated-h5ad", required=True, help="Generated h5ad with same var and obs.treatment_time")
    parser.add_argument("--train-h5ad", default=None, help="Train h5ad for control (0h) for delta metrics; optional")
    parser.add_argument("--time-key", default="treatment_time", help="obs column for time labels")
    parser.add_argument("--n-samples", type=int, default=0, help="Subsample per time point (0 = max available)")
    parser.add_argument("--csv", type=str, default=None, help="Append row(s) of metrics to this CSV")
    parser.add_argument("--method-name", type=str, default="", help="Method name for CSV row")
    parser.add_argument("--per-time", action="store_true", help="If set with --csv, write one row per treatment_time (4h, 6h) instead of one averaged row")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (overridden by RUN_SEED env per run)")
    args = parser.parse_args()

    from utils.seed import resolve_seed, set_seed
    set_seed(resolve_seed(getattr(args, "seed", 0)))
    run_seed = resolve_seed(getattr(args, "seed", 0))

    adata_test = sc.read_h5ad(args.test_h5ad)
    adata_gen = sc.read_h5ad(args.generated_h5ad)
    if args.time_key not in adata_test.obs.columns:
        raise KeyError(f"Test adata must have obs['{args.time_key}']")
    if args.time_key not in adata_gen.obs.columns:
        raise KeyError(f"Generated adata must have obs['{args.time_key}']")

    times_test = adata_test.obs[args.time_key].astype(str).str.strip().unique()
    times_gen = adata_gen.obs[args.time_key].astype(str).str.strip().unique()
    times = sorted(set(times_test) & set(times_gen))
    if not times:
        raise ValueError("No common treatment_time values between test and generated.")

    from utils.max_eval_samples import resolve_eval_n_samples
    if args.n_samples is None or args.n_samples <= 0:
        args.n_samples = resolve_eval_n_samples(args.test_h5ad, 0, mode="timepoint", time_col=args.time_key)
    print(f"Using n_samples={args.n_samples} per time point for evaluation.")

    # Control from train (0h) for delta metrics
    ctrl_pb = None
    if args.train_h5ad and os.path.isfile(args.train_h5ad):
        adata_train = sc.read_h5ad(args.train_h5ad)
        if args.time_key in adata_train.obs.columns:
            t0 = "0h"
            if t0 in adata_train.obs[args.time_key].astype(str).values:
                ctrl = adata_train[adata_train.obs[args.time_key].astype(str).str.strip() == t0]
                if ctrl.n_obs > 0:
                    ctrl_pb = np.mean(_toarray(ctrl.X), axis=0).astype(np.float64)

    n_samples = args.n_samples
    results = []
    for t in times:
        mask_t = adata_test.obs[args.time_key].astype(str).str.strip() == t
        mask_g = adata_gen.obs[args.time_key].astype(str).str.strip() == t
        real = adata_test[mask_t]
        gen = adata_gen[mask_g]
        if real.n_obs == 0 or gen.n_obs == 0:
            continue
        if n_samples:
            if real.n_obs >= n_samples:
                np.random.seed(run_seed)
                real = real[np.random.choice(real.n_obs, n_samples, replace=False)]
            if gen.n_obs >= n_samples:
                np.random.seed(run_seed)
                gen = gen[np.random.choice(gen.n_obs, n_samples, replace=False)]
        real_X = _toarray(real.X).astype(np.float64)
        gen_X = _toarray(gen.X).astype(np.float64)
        real_pb = np.mean(real_X, axis=0)
        pred_pb = np.mean(gen_X, axis=0)
        ctrl = ctrl_pb if ctrl_pb is not None else real_pb  # fallback

        mae = compute_mae(real_pb, pred_pb)
        r2 = compute_r2(real_X, gen_X)
        edist = compute_edistance(real_X, gen_X)
        mmd = compute_mmd(real_X, gen_X)
        p_all = compute_pearson(real_pb, pred_pb)
        pd_all = compute_pearson_delta(real_pb, pred_pb, ctrl)
        pd_de20 = compute_pearson_delta_de(real_pb, pred_pb, ctrl, k=20)
        pd_de50 = compute_pearson_delta_de(real_pb, pred_pb, ctrl, k=50)
        pd_de100 = compute_pearson_delta_de(real_pb, pred_pb, ctrl, k=100)

        delta_true = real_pb - ctrl
        de_idx = np.argsort(np.abs(delta_true))[::-1][:100]
        true_de = set(adata_test.var_names[de_idx].tolist())
        delta_pred = pred_pb - ctrl
        pred_de_idx = np.argsort(np.abs(delta_pred))[::-1][:100]
        pred_de = set(adata_test.var_names[pred_de_idx].tolist())
        pred_fc = {g: delta_pred[i] for i, g in enumerate(adata_test.var_names)}
        des = compute_des(true_de, pred_de, pred_fc)

        results.append({
            "time": t,
            "mae": mae, "r2": r2, "edistance": edist, "mmd": mmd,
            "pearson_all": p_all, "pearson_delta_all": pd_all,
            "pearson_delta_de20": pd_de20, "pearson_delta_de50": pd_de50, "pearson_delta_de100": pd_de100,
            "des": des,
        })

    if not results:
        print("No valid time groups.", file=sys.stderr)
        sys.exit(1)

    # PDS over time points (one pseudobulk per time)
    real_pbs = []
    pred_pbs = []
    for t in times:
        mask_t = adata_test.obs[args.time_key].astype(str).str.strip() == t
        mask_g = adata_gen.obs[args.time_key].astype(str).str.strip() == t
        real = adata_test[mask_t]
        gen = adata_gen[mask_g]
        if real.n_obs and gen.n_obs:
            real_pbs.append(np.mean(_toarray(real.X), axis=0))
            pred_pbs.append(np.mean(_toarray(gen.X), axis=0))
    pds_val = compute_pds(np.array(pred_pbs), np.array(real_pbs)) if len(pred_pbs) > 1 else 1.0

    # Averages
    n_res = len(results)
    mae_avg = np.mean([r["mae"] for r in results])
    des_avg = np.mean([r["des"] for r in results])
    edist_avg = np.mean([r["edistance"] for r in results])
    mmd_avg = np.mean([r["mmd"] for r in results])
    r2_avg = np.mean([r["r2"] for r in results])
    p_all_avg = np.mean([r["pearson_all"] for r in results])
    pd_all_avg = np.mean([r["pearson_delta_all"] for r in results])
    pd_de20_avg = np.mean([r["pearson_delta_de20"] for r in results])
    pd_de50_avg = np.mean([r["pearson_delta_de50"] for r in results])
    pd_de100_avg = np.mean([r["pearson_delta_de100"] for r in results])

    print("=" * 60)
    print("   Fig4 time-conditioned evaluation (averaged over time points)")
    print("=" * 60)
    print(f"Perturbation Discrimination Score (PDS): {pds_val:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_avg:.4f}")
    print(f"Differential Expression Score (DES): {des_avg:.4f}")
    print("-" * 20)
    print(f"E-Distance: {edist_avg:.4f}")
    print(f"Maximum Mean Discrepancy (MMD): {mmd_avg:.4f}")
    print(f"R-squared (R2): {r2_avg:.4f}")
    print("-" * 20)
    print(f"Pearson (all genes): {p_all_avg:.4f}")
    print(f"Pearson Delta (all genes): {pd_all_avg:.4f}")
    print(f"Pearson Delta (top 20 DE genes): {pd_de20_avg:.4f}")
    print(f"Pearson Delta (top 50 DE genes): {pd_de50_avg:.4f}")
    print(f"Pearson Delta (top 100 DE genes): {pd_de100_avg:.4f}")
    print("=" * 60)

    if args.csv:
        if getattr(args, "per_time", False) and results:
            # One row per treatment_time (4h, 6h)
            rows = []
            for r in results:
                rows.append({
                    "Method": args.method_name or "fig4",
                    "treatment_time": r["time"],
                    "PDS": pds_val,
                    "MAE": r["mae"], "DES": r["des"],
                    "E-Distance": r["edistance"], "MMD": r["mmd"], "R2": r["r2"],
                    "Pearson (all genes)": r["pearson_all"],
                    "Pearson Delta (all genes)": r["pearson_delta_all"],
                    "Pearson Delta (top 20 DE genes)": r["pearson_delta_de20"],
                    "Pearson Delta (top 50 DE genes)": r["pearson_delta_de50"],
                    "Pearson Delta (top 100 DE genes)": r["pearson_delta_de100"],
                })
            df = pd.DataFrame(rows)
        else:
            row = {
                "Method": args.method_name or "fig4",
                "PDS": pds_val, "MAE": mae_avg, "DES": des_avg,
                "E-Distance": edist_avg, "MMD": mmd_avg, "R2": r2_avg,
                "Pearson (all genes)": p_all_avg, "Pearson Delta (all genes)": pd_all_avg,
                "Pearson Delta (top 20 DE genes)": pd_de20_avg,
                "Pearson Delta (top 50 DE genes)": pd_de50_avg,
                "Pearson Delta (top 100 DE genes)": pd_de100_avg,
            }
            df = pd.DataFrame([row])
        write_header = not os.path.isfile(args.csv)
        df.to_csv(args.csv, mode="a", header=write_header, index=False)
        print(f"Appended {len(df)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
