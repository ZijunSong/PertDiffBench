#!/usr/bin/env python3
"""Merge CD4T gene-order supplementary metrics into one CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SAMPLE_ROOT = Path("/data/ppnm/data/PertDiffBench/samples/supp")
OUT_DIR = Path("/data/ppnm/PertDiffBench/supp")

# rel_path under samples/supp/{order}/CD4T/ -> canonical method name
METRIC_FILES = {
    "DDPM+MLP": "mlp_ddpm_mlp_1000/metrics_mlp_ddpm_mlp_CD4T_hvg_1000.csv",
    "DDPM": "scrna_ddpm_scrna_1000/metrics_ddpm_CD4T_hvg_1000.csv",
    "scGen": "scgen_1000/metrics_scGen_CD4T_hvg_1000.csv",
    "scDiff": "scdiff_1000/metrics_scDiff_CD4T_hvg_1000.csv",
    "Squidiff": "squidiff_1000/metrics_Squidiff_CD4T_hvg_1000.csv",
    "scDiffusion": "scDiffusion_1000/metrics_scDiffusion_CD4T_hvg_1000.csv",
}

KEY_METRICS = [
    "PDS (mean±std)",
    "MAE (mean±std)",
    "DES (mean±std)",
    "E-Distance (mean±std)",
    "MMD (mean±std)",
    "R2 (mean±std)",
    "Pearson (all genes) (mean±std)",
    "Pearson Delta (all genes) (mean±std)",
    "Pearson Delta (top 20 DE genes) (mean±std)",
    "Pearson Delta (top 50 DE genes) (mean±std)",
    "Pearson Delta (top 100 DE genes) (mean±std)",
]


def load_order(order: str) -> list[dict]:
    rows = []
    for method, rel in METRIC_FILES.items():
        path = SAMPLE_ROOT / order / "CD4T" / rel
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()
        row["gene_order"] = order
        row["Method"] = method
        row["cell_type"] = "CD4T"
        row["task"] = "fig1_task1_known_condition"
        row["n_genes"] = 1000
        row["source_csv"] = str(path)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["hvg_rank", "shuffle", "cluster"],
        help="Gene order conditions to merge (skip missing).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "gene_order_cd4t_metrics_merged.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=OUT_DIR / "gene_order_cd4t_metrics_summary.csv",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for order in args.orders:
        all_rows.extend(load_order(order))

    if not all_rows:
        raise SystemExit("No metric CSV found.")

    df = pd.DataFrame(all_rows)
    front = ["gene_order", "Method", "cell_type", "task", "n_genes"]
    rest = [c for c in df.columns if c not in front + ["source_csv"]]
    df = df[front + rest + ["source_csv"]]
    df = df.sort_values(["Method", "gene_order"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")

    summary_cols = [c for c in KEY_METRICS if c in df.columns]
    summary = df[front[:2] + summary_cols].copy()
    summary.to_csv(args.summary_out, index=False)
    print(f"Wrote summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
