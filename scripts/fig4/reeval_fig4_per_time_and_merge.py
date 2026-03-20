#!/usr/bin/env python3
"""
基于已有 synthetic_fig4.h5ad 重新跑评估，按 4h/6h 分别输出（每方法两行），再汇总为一个大 CSV。
不依赖训练，只需 samples/fig4/<method>/run*/synthetic_fig4.h5ad 存在。
输出：samples/fig4/fig4_metrics_merged.csv（列含 Method, treatment_time, 及各指标 mean±std）。
"""
import os
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SAMPLES_FIG4 = REPO_ROOT / "samples" / "fig4"
FIG4_TASK1_DIR = Path("/data/ppnm/data/PertDiffBench/data/fig4_task1")
TEST_H5 = FIG4_TASK1_DIR / "fig4_test.h5ad"
TRAIN_H5 = FIG4_TASK1_DIR / "fig4_train.h5ad"
OUT_CSV = SAMPLES_FIG4 / "fig4_metrics_merged.csv"

# 子目录名 -> 显示用的 Method 名
METHOD_DIR_TO_NAME = {
    "scDiffusion_3000": "scDiffusion",
    "scrna_ddpm_scrna_3000": "DDPM",
    "mlp_ddpm_mlp_3000": "DDPM+MLP",
    "squidiff_3000": "Squidiff",
}


def main():
    if not SAMPLES_FIG4.is_dir():
        raise SystemExit(f"目录不存在: {SAMPLES_FIG4}")
    if not TEST_H5.is_file():
        raise SystemExit(f"测试集不存在: {TEST_H5}")
    if not TRAIN_H5.is_file():
        raise SystemExit(f"训练集不存在: {TRAIN_H5}")

    eval_script = REPO_ROOT / "scripts" / "fig4" / "eval_fig4_time_conditioned.py"
    if not eval_script.is_file():
        raise SystemExit(f"评估脚本不存在: {eval_script}")

    metric_cols = [
        "PDS", "MAE", "DES", "E-Distance", "MMD", "R2",
        "Pearson (all genes)", "Pearson Delta (all genes)",
        "Pearson Delta (top 20 DE genes)", "Pearson Delta (top 50 DE genes)",
        "Pearson Delta (top 100 DE genes)",
    ]

    all_rows = []
    for subdir in sorted(SAMPLES_FIG4.iterdir()):
        if not subdir.is_dir():
            continue
        method_name = METHOD_DIR_TO_NAME.get(subdir.name, subdir.name)
        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run"):
                continue
            h5ad = run_dir / "synthetic_fig4.h5ad"
            if not h5ad.is_file():
                continue
            # 使用尚未存在的路径，否则评估脚本会认为文件已存在而不写表头
            tmp_csv = os.path.join(tempfile.gettempdir(), f"fig4_reeval_{uuid.uuid4().hex}.csv")
            try:
                cmd = [
                    sys.executable,
                    str(eval_script),
                    "--test-h5ad", str(TEST_H5),
                    "--generated-h5ad", str(h5ad),
                    "--train-h5ad", str(TRAIN_H5),
                    "--per-time", "--csv", tmp_csv,
                    "--method-name", method_name,
                ]
                ret = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
                if ret.returncode != 0:
                    if ret.stderr:
                        print(ret.stderr, file=sys.stderr)
                    if ret.stdout:
                        print(ret.stdout, file=sys.stderr)
                    ret.check_returncode()
                df = pd.read_csv(tmp_csv)
                df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
                df["_run"] = run_dir.name
                all_rows.append(df)
            except Exception as e:
                print(f"[WARN] 跳过 {h5ad}: {e}")
            finally:
                try:
                    os.unlink(tmp_csv)
                except Exception:
                    pass

    if not all_rows:
        print("未找到任何 run*/synthetic_fig4.h5ad，请先完成各 baseline 的 fig4 生成。")
        pd.DataFrame(columns=["Method", "treatment_time"] + metric_cols).to_csv(OUT_CSV, index=False)
        print(f"已创建空文件: {OUT_CSV}")
        return

    big = pd.concat(all_rows, ignore_index=True)
    big.columns = big.columns.str.strip().str.replace("\ufeff", "", regex=False)
    # 兼容列名大小写/空格
    method_col = "Method" if "Method" in big.columns else next((c for c in big.columns if c.lower() == "method"), None)
    time_col = "treatment_time" if "treatment_time" in big.columns else next((c for c in big.columns if "treatment" in c.lower() and "time" in c.lower()), None)
    if method_col is None or time_col is None:
        raise SystemExit(f"CSV 缺少分组列。当前列: {list(big.columns)}")
    if method_col != "Method":
        big = big.rename(columns={method_col: "Method"})
    if time_col != "treatment_time":
        big = big.rename(columns={time_col: "treatment_time"})
    # 按 (Method, treatment_time) 聚合并计算 mean ± std
    out_rows = []
    for (method, tt), g in big.groupby(["Method", "treatment_time"]):
        row = {"Method": method, "treatment_time": tt}
        for c in metric_cols:
            if c not in g.columns:
                continue
            vals = g[c].astype(float)
            mu, std = vals.mean(), vals.std()
            if pd.isna(std) or std == 0:
                std = 0.0
            row[c] = f"{mu:.4f}±{std:.4f}"
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    # 列顺序：Method, treatment_time, 再按 metric_cols
    cols = ["Method", "treatment_time"] + [c for c in metric_cols if c in out_df.columns]
    out_df = out_df[[c for c in cols if c in out_df.columns]]
    out_df.to_csv(OUT_CSV, index=False)
    print(f"已按 4h/6h 汇总，共 {len(out_df)} 行 -> {OUT_CSV}")
    print(f"绝对路径: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
