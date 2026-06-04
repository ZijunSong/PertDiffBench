#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import anndata as ad
import pandas as pd

# Processed data root (script can be run from repo root)
DATA_OUT = Path("/data/ppnm/data/PertDiffBench/data/fig2_task1_unseenMOA")

STATUS_COL = "perturbation_status"

# ------------------------- Auto NUM_GENES detection -------------------------
def detect_num_genes(root: Path) -> int | None:
    """
    Detect gene number from IFN-only h5ad files.
    Ensures all datasets share the same n_vars.
    """
    gene_counts = set()

    for split in ["unseen_diff_moa", "unseen_same_moa"]:
        h5ad_dir = root / split / "h5ad"
        if not h5ad_dir.exists():
            continue

        for fp in h5ad_dir.glob("*.h5ad"):
            adata = ad.read_h5ad(fp)
            gene_counts.add(int(adata.n_vars))

    if not gene_counts:
        print("[WARN] No IFN-only h5ad found for gene detection.")
        return None

    if len(gene_counts) > 1:
        print(f"[ERROR] Inconsistent gene numbers detected across datasets: {gene_counts}")
        print("You cannot set a single NUM_GENES safely.")
        return None

    num_genes = gene_counts.pop()
    print(f"[INFO] Detected consistent gene number across datasets: NUM_GENES = {num_genes}")
    return num_genes

def find_root(start: Path) -> Path:
    """
    Auto-detect task1_unseenMOA root by walking upwards.
    Root is identified by presence of:
      - unseen_diff_moa/ and unseen_same_moa/
    """
    cur = start.resolve()
    for _ in range(8):
        if (cur / "unseen_diff_moa").is_dir() and (cur / "unseen_same_moa").is_dir():
            return cur
        cur = cur.parent
    raise FileNotFoundError(
        f"Cannot locate root containing 'unseen_diff_moa' and 'unseen_same_moa' from: {start}"
    )

def summarize_one_h5ad(fp: Path, col: str = STATUS_COL) -> dict:
    info = {
        "file": fp.name,
        "path": str(fp),
        "n_obs": None,
        "n_vars": None,
        "has_col": False,
        "count_Control": 0,
        "count_IFN": 0,
        "count_other": 0,
        "other_values": {},
    }
    adata = ad.read_h5ad(fp)
    info["n_obs"] = int(adata.n_obs)
    info["n_vars"] = int(adata.n_vars)

    if col not in adata.obs.columns:
        return info

    info["has_col"] = True
    s = adata.obs[col].astype(str)
    vc = s.value_counts(dropna=False)

    info["count_Control"] = int(vc.get("Control", 0))
    info["count_IFN"] = int(vc.get("IFN", 0))

    other = vc.drop(labels=[x for x in ["Control", "IFN"] if x in vc.index], errors="ignore")
    info["count_other"] = int(other.sum()) if len(other) else 0
    info["other_values"] = dict(other) if len(other) else {}

    return info

def load_control_nobs(root: Path) -> int | None:
    """
    Optional: if control_merged.h5ad exists, read and return its n_obs.
    """
    fp = root / "control_merged.h5ad"
    if not fp.exists():
        return None
    adata = ad.read_h5ad(fp)
    return int(adata.n_obs)

def check_one_split(root: Path, split: str) -> None:
    """
    split: "unseen_diff_moa" or "unseen_same_moa"
    """
    ifn_dir = root / split / "h5ad"
    comb_dir = root / "control_plus_ifn" / split

    print(f"\n==================== {split} ====================")
    print(f"IFN-only dir  : {ifn_dir}")
    print(f"Combined dir  : {comb_dir}")

    if not ifn_dir.exists():
        print(f"[WARN] IFN-only dir missing: {ifn_dir}")
        return

    ifn_files = sorted(ifn_dir.glob("*.h5ad"))
    if not ifn_files:
        print(f"[WARN] No IFN-only .h5ad files in: {ifn_dir}")
        return

    # 1) Summarize IFN-only
    rows = [summarize_one_h5ad(fp, col=STATUS_COL) for fp in ifn_files]
    df = pd.DataFrame(rows)

    show_cols = ["file", "n_obs", "n_vars", "has_col", "count_Control", "count_IFN", "count_other"]
    print("\n=== IFN-only files: perturbation_status check ===")
    print(df[show_cols].to_string(index=False))

    missing = df[~df["has_col"]]
    if len(missing):
        print("\n[WARN] IFN-only files missing 'perturbation_status':")
        print(missing["file"].to_string(index=False))

    weird = df[(df["has_col"]) & (df["count_other"] > 0)]
    if len(weird):
        print("\n[WARN] IFN-only files with unexpected perturbation_status values (not Control/IFN):")
        for _, r in weird.iterrows():
            print(f"  - {r['file']}: other_values={r['other_values']}")

    # 2) Pairing check with combined
    if not comb_dir.exists():
        print("\n[NOTE] Combined dir missing; cannot do pairing check.")
        return

    comb_files = sorted(comb_dir.glob("*.h5ad"))
    comb_map = {fp.name: fp for fp in comb_files}

    print(f"\nCombined h5ad files found: {len(comb_files)}")
    print("=== Pairing check: expected <IFN_STEM>__plus_control.h5ad ===")

    control_n = load_control_nobs(root)

    pair_rows = []
    for ifn_fp in ifn_files:
        stem = ifn_fp.stem
        expected = f"{stem}__plus_control.h5ad"
        comb_fp = comb_map.get(expected)

        # IFN-only n_obs for later sanity check
        ifn_info = summarize_one_h5ad(ifn_fp, col=STATUS_COL)
        ifn_n = ifn_info["n_obs"]

        rec = {
            "ifn_file": ifn_fp.name,
            "ifn_n_obs": ifn_n,
            "expected_combined": expected,
            "combined_exists": comb_fp is not None,
            "combined_n_obs": None,
            "combined_count_Control": None,
            "combined_count_IFN": None,
            "combined_has_both": None,
            "nobs_matches_control_plus_ifn": None,
        }

        if comb_fp is not None:
            comb_info = summarize_one_h5ad(comb_fp, col=STATUS_COL)
            rec["combined_n_obs"] = comb_info["n_obs"]
            rec["combined_count_Control"] = comb_info["count_Control"] if comb_info["has_col"] else None
            rec["combined_count_IFN"] = comb_info["count_IFN"] if comb_info["has_col"] else None
            rec["combined_has_both"] = (
                comb_info["has_col"] and comb_info["count_Control"] > 0 and comb_info["count_IFN"] > 0
            )

            # Strong sanity check: combined n_obs should equal control_n + ifn_n (if control_n known)
            if control_n is not None:
                rec["nobs_matches_control_plus_ifn"] = (comb_info["n_obs"] == (control_n + ifn_n))

        pair_rows.append(rec)

    dfp = pd.DataFrame(pair_rows)
    # Display more columns if control_merged.h5ad exists
    base_cols = [
        "ifn_file", "ifn_n_obs", "combined_exists", "combined_n_obs",
        "combined_count_Control", "combined_count_IFN", "combined_has_both"
    ]
    if control_n is not None:
        base_cols.append("nobs_matches_control_plus_ifn")

    print(dfp[base_cols].to_string(index=False))

    miss_pair = dfp[~dfp["combined_exists"]]
    if len(miss_pair):
        print("\n[WARN] Missing combined files for some IFN datasets:")
        print(miss_pair[["ifn_file", "expected_combined"]].to_string(index=False))

    bad_comb = dfp[(dfp["combined_exists"]) & (dfp["combined_has_both"] == False)]
    if len(bad_comb):
        print("\n[WARN] Combined files exist but do NOT contain both Control and IFN:")
        print(bad_comb[["ifn_file", "expected_combined", "combined_count_Control", "combined_count_IFN"]].to_string(index=False))

    if control_n is not None:
        mismatch = dfp[(dfp["combined_exists"]) & (dfp["nobs_matches_control_plus_ifn"] == False)]
        if len(mismatch):
            print(f"\n[WARN] n_obs mismatch against control_merged.h5ad (control_n_obs={control_n}).")
            print("This suggests rows were dropped/duplicated during merge or Control used is not control_merged.h5ad.")
            print(mismatch[["ifn_file", "ifn_n_obs", "combined_n_obs", "nobs_matches_control_plus_ifn"]].to_string(index=False))

def main():
    start = Path(os.getcwd())
    root = DATA_OUT
    print(f"Working dir: {start}")
    print(f"Data root: {root}")

    detected_genes = detect_num_genes(root)
    if detected_genes is not None:
        print(f"[SUGGESTION] You should set in your .sh script:")
        print(f'  NUM_GENES="{detected_genes}"')

    ctrl_fp = root / "control_merged.h5ad"
    if ctrl_fp.exists():
        ctrl = ad.read_h5ad(ctrl_fp)
        s = ctrl.obs[STATUS_COL].astype(str) if STATUS_COL in ctrl.obs.columns else None
        ctrl_status = dict(s.value_counts()) if s is not None else None
        print(f"Control file: {ctrl_fp} (n_obs={ctrl.n_obs}, n_vars={ctrl.n_vars}, status_counts={ctrl_status})")
    else:
        print("[NOTE] control_merged.h5ad not found at root; n_obs strong-check will be skipped.")

    check_one_split(root, "unseen_diff_moa")
    check_one_split(root, "unseen_same_moa")
    print("\nDone.")

if __name__ == "__main__":
    main()
