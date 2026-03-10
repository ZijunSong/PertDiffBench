#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import anndata as ad

# 原始数据与处理后数据的根路径（可从项目根目录运行脚本）
DATA_ORI = Path("/data/ppnm/data/PertDiffBench/data_ori/fig2/task1_unseenMOA")
DATA_OUT = Path("/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA")

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def build_control_h5ad_from_csv(
    exp_csv: Path,
    meta_csv: Path,
    out_h5ad: Optional[Path] = None,
) -> ad.AnnData:
    df_exp = pd.read_csv(exp_csv, index_col=0, low_memory=False)
    df_meta = pd.read_csv(meta_csv, index_col=0, low_memory=False)

    df_exp.index = df_exp.index.astype(str)
    df_meta.index = df_meta.index.astype(str)

    if not df_exp.index.equals(df_meta.index):
        common = df_exp.index.intersection(df_meta.index)
        if len(common) == 0:
            raise ValueError(f"Control exp/meta index has no overlap: {exp_csv} vs {meta_csv}")
        df_exp = df_exp.loc[common, :]
        df_meta = df_meta.loc[common, :]

    adata = ad.AnnData(
        X=df_exp.to_numpy(),
        obs=df_meta.copy(),
        var=pd.DataFrame(index=pd.Index(df_exp.columns.astype(str), name="gene")),
    )
    adata.obs["perturbation_status"] = "Control"
    adata.obs["source_dataset"] = "Control"

    if out_h5ad is not None:
        adata.write(out_h5ad)

    return adata


def load_or_make_control(data_ori: Path, data_out: Path) -> ad.AnnData:
    control_h5ad = data_out / "control_merged.h5ad"
    if control_h5ad.exists():
        adata = ad.read_h5ad(control_h5ad)
        if "perturbation_status" not in adata.obs:
            adata.obs["perturbation_status"] = "Control"
        adata.obs["source_dataset"] = "Control"
        return adata

    exp_csv = data_ori / "dfcontrol_exp.csv"
    meta_csv = data_ori / "dfcontrol_meta.csv"
    if not exp_csv.exists() or not meta_csv.exists():
        raise FileNotFoundError(
            f"Neither {control_h5ad} exists, nor {exp_csv}/{meta_csv} found."
        )

    data_out.mkdir(parents=True, exist_ok=True)
    return build_control_h5ad_from_csv(exp_csv, meta_csv, out_h5ad=control_h5ad)


def _ensure_sparse(adata: ad.AnnData) -> ad.AnnData:
    if sp is None:
        return adata
    if sp.issparse(adata.X):
        return adata
    adata.X = sp.csr_matrix(adata.X)
    return adata


def merge_control_with_ifn(control: ad.AnnData, ifn: ad.AnnData, ifn_tag: str) -> ad.AnnData:
    c = control.copy()
    i = ifn.copy()

    c.obs["perturbation_status"] = "Control"
    c.obs["source_dataset"] = "Control"

    # 你的 IFN 数据之前已经写了 perturbation_status=IFN；这里强制一遍也无妨
    i.obs["perturbation_status"] = "IFN"
    i.obs["source_dataset"] = ifn_tag

    # Make obs_names unique
    c.obs_names = pd.Index([f"Control::{x}" for x in c.obs_names.astype(str)], dtype=str)
    i.obs_names = pd.Index([f"{ifn_tag}::{x}" for x in i.obs_names.astype(str)], dtype=str)

    c = _ensure_sparse(c)
    i = _ensure_sparse(i)

    # ✅ 关键修复：不要同时“用 dict 的 key”又传 keys=
    # 这里改用 list + keys 的方式（更直观也更兼容）
    try:
        merged = ad.concat(
            [c, i],
            axis=0,
            join="outer",
            keys=["control", "ifn"],   # 只在这里指定类别一次
            label="concat_batch",
            fill_value=0,
            index_unique=None,
        )
    except TypeError:
        # 兼容旧版 anndata：可能不支持 fill_value
        merged = ad.concat(
            [c, i],
            axis=0,
            join="outer",
            keys=["control", "ifn"],
            label="concat_batch",
            index_unique=None,
        )
        if sp is None or not sp.issparse(merged.X):
            import numpy as np
            merged.X = np.nan_to_num(merged.X, nan=0.0)

    merged.uns["merged_from"] = {"control": "control_merged.h5ad", "ifn_tag": ifn_tag}
    return merged


def process_ifn_folder(data_out: Path, subdir: str, control: ad.AnnData) -> None:
    h5ad_dir = data_out / subdir / "h5ad"
    if not h5ad_dir.exists():
        print(f"[SKIP] {h5ad_dir} not found.")
        return

    out_root = data_out / "control_plus_ifn" / subdir
    out_root.mkdir(parents=True, exist_ok=True)

    ifn_files = sorted(h5ad_dir.glob("*.h5ad"))
    if not ifn_files:
        print(f"[WARN] No .h5ad files in {h5ad_dir}")
        return

    print(f"\n=== Merging Control with IFN files in: {h5ad_dir} ===")

    for fp in ifn_files:
        ifn_tag = fp.stem
        out_path = out_root / f"{fp.stem}__plus_control.h5ad"

        if out_path.exists():
            print(f"[OK] Exists, skip: {out_path}")
            continue

        try:
            ifn = ad.read_h5ad(fp)
            merged = merge_control_with_ifn(control, ifn, ifn_tag=ifn_tag)
            merged.write(out_path)
            print(f"[OK] Wrote: {out_path}  (cells={merged.n_obs}, genes={merged.n_vars})")
        except Exception as e:
            print(f"[ERR] Failed on {fp}: {e}")


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    control = load_or_make_control(DATA_ORI, DATA_OUT)

    process_ifn_folder(DATA_OUT, "unseen_diff_moa", control)
    process_ifn_folder(DATA_OUT, "unseen_same_moa", control)

    print("\nDone.")


if __name__ == "__main__":
    main()
