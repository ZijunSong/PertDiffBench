#!/usr/bin/env python3
"""
汇总 fig4 各 baseline（scDiffusion, DDPM, DDPM+MLP, Squidiff）实验生成的 metrics CSV 为一个大 CSV。
各 fig4_task1_*.sh 将结果写入 samples/fig4/<method_dir>/metrics_*.csv，
本脚本合并所有 metrics_*.csv，输出写入 samples/fig4/fig4_metrics_merged.csv。
"""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_FIG4 = REPO_ROOT / "samples" / "fig4"
OUT_CSV = SAMPLES_FIG4 / "fig4_metrics_merged.csv"


def main():
    if not SAMPLES_FIG4.is_dir():
        raise SystemExit(f"目录不存在: {SAMPLES_FIG4}")

    # 收集所有 samples/fig4/*/metrics_*.csv
    csv_paths = sorted(SAMPLES_FIG4.glob("*/metrics_*.csv"))
    csv_paths = [p for p in csv_paths if p.is_file()]

    if not csv_paths:
        print("未找到任何 fig4 metrics CSV，请先运行各 baseline 的 fig4_task1_*.sh。")
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        print(f"已创建空文件: {OUT_CSV}")
        return

    all_dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] 跳过 {p}: {e}")

    if not all_dfs:
        print("没有可合并的 CSV 内容。")
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        print(f"已创建空文件: {OUT_CSV}")
        return

    merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    merged.to_csv(OUT_CSV, index=False)
    print(f"已汇总 {len(all_dfs)} 个 CSV，共 {len(merged)} 行 -> {OUT_CSV}")
    print(f"绝对路径: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
