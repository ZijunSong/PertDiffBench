#!/usr/bin/env python3
"""Convert PertDiffBench MOA train/test h5ad into a single ChemCPA-compatible h5ad.

Adds columns required by chemCPA (https://github.com/theislab/chemCPA):
  control, split, perturbation, dose_value, smiles, celltype, cov_drug_dose_name
and ``uns['all_DEGs']`` for differential-expression supervision.
"""

from __future__ import annotations

import argparse
import anndata as ad
import numpy as np
import pandas as pd


def _ensure_celltype(obs: pd.DataFrame, default: str = "all") -> pd.Series:
    if "celltype" in obs.columns:
        return obs["celltype"].astype(str)
    return pd.Series([default] * len(obs), index=obs.index)


def _pert_name(row: pd.Series, drug_key: str) -> str:
    if str(row.get("perturbation_status", "")).strip() == "Control":
        return "control"
    val = str(row.get(drug_key, "")).strip()
    return val if val and val.lower() not in {"nan", "none", ""} else "unknown_drug"


def _dose_val(row: pd.Series, dose_key: str) -> float:
    if str(row.get("perturbation_status", "")).strip() == "Control":
        return 0.0
    try:
        return float(row.get(dose_key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _smiles_val(row: pd.Series, smiles_key: str) -> str:
    if str(row.get("perturbation_status", "")).strip() == "Control":
        return ""
    val = row.get(smiles_key, "")
    if pd.isna(val):
        return ""
    return str(val).strip()


def _annotate_obs(
    adata: ad.AnnData,
    split: str,
    drug_key: str,
    dose_key: str,
    smiles_key: str,
) -> ad.AnnData:
    out = adata.copy()
    obs = out.obs.copy()
    celltype = _ensure_celltype(obs)
    perturbation = obs.apply(lambda r: _pert_name(r, drug_key), axis=1)
    dose = obs.apply(lambda r: _dose_val(r, dose_key), axis=1)
    smiles = obs.apply(lambda r: _smiles_val(r, smiles_key), axis=1)
    is_control = (obs["perturbation_status"].astype(str) == "Control").astype(int)

    obs["celltype"] = celltype.values
    obs["perturbation"] = perturbation.values
    obs[dose_key] = dose.values
    obs[smiles_key] = smiles.values
    obs["control"] = is_control.values
    obs["split"] = split
    obs["cov_drug_dose_name"] = [
        f"{ct}_{pert}_{dose_v}" for ct, pert, dose_v in zip(celltype, perturbation, dose)
    ]
    out.obs = obs
    return out


def _compute_degs(
    adata: ad.AnnData,
    pert_category: str = "cov_drug_dose_name",
    top_n: int = 100,
) -> dict[str, list[str]]:
    """Top-|delta| genes per perturbation category vs pooled control."""
    ctrl_mask = adata.obs["control"].astype(int) == 1
    if not ctrl_mask.any():
        raise ValueError("No control cells found when computing DEGs.")

    ctrl_mean = np.asarray(adata[ctrl_mask].X.mean(axis=0)).ravel()
    var_names = adata.var_names.astype(str).tolist()
    degs: dict[str, list[str]] = {}

    for cat in adata.obs[pert_category].astype(str).unique():
        treat_mask = (adata.obs[pert_category].astype(str) == cat) & (~ctrl_mask)
        if not treat_mask.any():
            degs[cat] = []
            continue
        treat_mean = np.asarray(adata[treat_mask].X.mean(axis=0)).ravel()
        delta = treat_mean - ctrl_mean
        idx = np.argsort(np.abs(delta))[::-1][:top_n]
        degs[cat] = [var_names[i] for i in idx]
    return degs


def build_chemcpa_h5ad(
    train_path: str,
    test_path: str,
    output_path: str,
    drug_key: str = "perturbation",
    dose_key: str = "dose_value",
    smiles_key: str = "smiles",
    degs_key: str = "all_DEGs",
) -> ad.AnnData:
    train = ad.read_h5ad(train_path)
    test = ad.read_h5ad(test_path)

    train = _annotate_obs(train, split="train", drug_key=drug_key, dose_key=dose_key, smiles_key=smiles_key)
    test = _annotate_obs(test, split="test", drug_key=drug_key, dose_key=dose_key, smiles_key=smiles_key)

    combined = ad.concat([train, test], join="outer", index_unique="-", fill_value=0)
    combined.uns[degs_key] = _compute_degs(combined)
    combined.write_h5ad(output_path)
    return combined


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-path", required=True)
    p.add_argument("--test-path", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--drug-key", default="perturbation")
    p.add_argument("--dose-key", default="dose_value")
    p.add_argument("--smiles-key", default="smiles")
    p.add_argument("--degs-key", default="all_DEGs")
    args = p.parse_args()

    adata = build_chemcpa_h5ad(
        args.train_path,
        args.test_path,
        args.output,
        drug_key=args.drug_key,
        dose_key=args.dose_key,
        smiles_key=args.smiles_key,
        degs_key=args.degs_key,
    )
    n_train = int((adata.obs["split"] == "train").sum())
    n_test = int((adata.obs["split"] == "test").sum())
    print(f"Wrote ChemCPA h5ad: {args.output} (cells={adata.n_obs}, genes={adata.n_vars}, train={n_train}, test={n_test})")


if __name__ == "__main__":
    main()
