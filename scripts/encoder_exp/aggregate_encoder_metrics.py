#!/usr/bin/env python3
"""
 encoder_exp under .sh metrics CSV as CSV.
 cellfm ( ).output samples/encoder_exp/encoder_exp_metrics_merged.csv
"""
from pathlib import Path
import pandas as pd

# indirectory: scripts/encoder_exp, directoryas on level
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_ENCODER = REPO_ROOT / "samples" / "encoder_exp"

# and subdir ( with cellfm_ddpm); eachsubdirunder metrics CSV glob 
# as samples/encoder_exp undersubdirname, valueas directoryunder metrics file glob
ENCODER_CSV_SPEC = [
    ("scvi_ddpm", "metrics_*.csv"),
    ("scimilarity_ddpm", "metrics_*.csv"),
    ("state_ddpm", "metrics_*.csv"),
    ("geneformer_ddpm", "metrics_*.csv"), # only directoryunder metrics_*.csv, with encoder/embeddings/
    ("scgpt_ddpm", "metrics_*.csv"),
    ("scfoundation_ddpm", "metrics_*.csv"),
    ("tx1_ddpm", "tx1_ddpm_*.csv"),
]


def main():
    if not SAMPLES_ENCODER.is_dir():
        raise SystemExit(f"Directory does not exist: {SAMPLES_ENCODER}")

    all_dfs = []
    for subdir_name, pattern in ENCODER_CSV_SPEC:
        subdir = SAMPLES_ENCODER / subdir_name
        if not subdir.is_dir():
            continue
        for csv_path in sorted(subdir.glob(pattern)):
            if not csv_path.is_file():
                continue
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                all_dfs.append(df)
            except Exception as e:
                print(f"[WARN] skip {csv_path}: {e}")

    if not all_dfs:
        print(" found metrics CSV, run first: encoder .sh .")
        out_path = SAMPLES_ENCODER / "encoder_exp_metrics_merged.csv"
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"created empty file: {out_path}")
        return

    merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    out_path = SAMPLES_ENCODER / "encoder_exp_metrics_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f" {len(all_dfs)} CSV, {len(merged)} -> {out_path}")
    print(f"absolute path: {out_path.resolve()}")


if __name__ == "__main__":
    main()
