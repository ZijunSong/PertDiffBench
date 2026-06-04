#!/usr/bin/env python3
"""
 fig4 baseline (scDiffusion, DDPM, DDPM+MLP, Squidiff) metrics CSV as CSV.
 fig4_task1_*.sh results samples/fig4/<method_dir>/metrics_*.csv, 
 andall metrics_*.csv, output samples/fig4/fig4_metrics_merged.csv.
"""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_FIG4 = REPO_ROOT / "samples" / "fig4"
OUT_CSV = SAMPLES_FIG4 / "fig4_metrics_merged.csv"


def main():
    if not SAMPLES_FIG4.is_dir():
        raise SystemExit(f"Directory does not exist: {SAMPLES_FIG4}")

    # all samples/fig4/*/metrics_*.csv
    csv_paths = sorted(SAMPLES_FIG4.glob("*/metrics_*.csv"))
    csv_paths = [p for p in csv_paths if p.is_file()]

    if not csv_paths:
        print(" found fig4 metrics CSV, run first: baseline fig4_task1_*.sh.")
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        print(f"created empty file: {OUT_CSV}")
        return

    all_dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    if not all_dfs:
        print("nocan and CSV inside .")
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        print(f"created empty file: {OUT_CSV}")
        return

    merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    merged.to_csv(OUT_CSV, index=False)
    print(f" {len(all_dfs)} CSV, {len(merged)} -> {OUT_CSV}")
    print(f"absolute path: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
