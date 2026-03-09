#!/usr/bin/env python3
"""
汇总 encoder_exp 下各 .sh 实验生成的 metrics CSV 为一个大 CSV。
排除 cellfm（已废弃）。输出写入 samples/encoder_exp/encoder_exp_metrics_merged.csv
"""
from pathlib import Path
import pandas as pd

# 脚本所在目录: scripts/encoder_exp，仓库根目录为其上两级
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_ENCODER = REPO_ROOT / "samples" / "encoder_exp"

# 参与汇总的子目录（不含 cellfm_ddpm）；每个子目录下 metrics CSV 的 glob 模式
# 键为 samples/encoder_exp 下的子目录名，值为该目录下 metrics 文件的 glob
ENCODER_CSV_SPEC = [
    ("scvi_ddpm", "metrics_*.csv"),
    ("scimilarity_ddpm", "metrics_*.csv"),
    ("state_ddpm", "metrics_*.csv"),
    ("geneformer_ddpm", "metrics_*.csv"),   # 仅根目录下 metrics_*.csv，不含 encoder/embeddings/
    ("scgpt_ddpm", "metrics_*.csv"),
    ("scfoundation_ddpm", "metrics_*.csv"),
    ("tx1_ddpm", "tx1_ddpm_*.csv"),
]


def main():
    if not SAMPLES_ENCODER.is_dir():
        raise SystemExit(f"目录不存在: {SAMPLES_ENCODER}")

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
                print(f"[WARN] 跳过 {csv_path}: {e}")

    if not all_dfs:
        print("未找到任何 metrics CSV，请先运行各 encoder 的 .sh 实验。")
        out_path = SAMPLES_ENCODER / "encoder_exp_metrics_merged.csv"
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"已创建空文件: {out_path}")
        return

    merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    out_path = SAMPLES_ENCODER / "encoder_exp_metrics_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"已汇总 {len(all_dfs)} 个 CSV，共 {len(merged)} 行 -> {out_path}")
    print(f"绝对路径: {out_path.resolve()}")


if __name__ == "__main__":
    main()
