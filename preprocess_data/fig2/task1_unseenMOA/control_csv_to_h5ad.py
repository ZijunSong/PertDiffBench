#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge control expression and metadata CSV files into a single AnnData object,
add a 'perturbation_status' column, and save as h5ad.
"""

from pathlib import Path

import pandas as pd
import anndata as ad

# 原始数据与处理后数据的根路径（可从项目根目录运行脚本）
DATA_ORI = Path("/data/ppnm/data/PertDiffBench/data_ori/fig2/task1_unseenMOA")
DATA_OUT = Path("/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA")

# =========================
# 1. Load CSV files
# =========================
exp_path = DATA_ORI / "dfcontrol_exp.csv"
meta_path = DATA_ORI / "dfcontrol_meta.csv"

print("Loading expression data...")
df_exp = pd.read_csv(exp_path, index_col=0)

print("Loading metadata...")
df_meta = pd.read_csv(meta_path, index_col=0)

# =========================
# 2. Sanity check alignment
# =========================
if not df_exp.index.equals(df_meta.index):
    raise ValueError("Index mismatch between expression and metadata!")

print("Index aligned. Creating AnnData object...")

# =========================
# 3. Create AnnData
# =========================
adata = ad.AnnData(
    X=df_exp.values,
    obs=df_meta.copy(),
    var=pd.DataFrame(index=df_exp.columns)
)

# =========================
# 4. Add perturbation_status column
# =========================
adata.obs["perturbation_status"] = "Control"

# =========================
# 5. Save to h5ad
# =========================
DATA_OUT.mkdir(parents=True, exist_ok=True)
output_path = DATA_OUT / "control_merged.h5ad"
adata.write(output_path)

print(f"Saved to {output_path}")
