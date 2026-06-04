#!/usr/bin/env python3
"""
 (leave-one-out) : types control_ifn.h5ad andas train set h5ad.
for types : test types , types and astraindata.

using :
  python merge_species_control_ifn.py \\
    --data-root data/fig2/task3_cross_species \\
    --train-species mouse,pig,rabbit \\
    --out data/fig2/task3_cross_species/merged_train_rat.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple species control_ifn h5ad files for leave-one-out training."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/fig2/task3_cross_species",
        help="Directory containing {species}_control_ifn.h5ad files",
    )
    parser.add_argument(
        "--train-species",
        type=str,
        required=True,
        help="Comma-separated list of species to merge as training data, e.g. mouse,pig,rabbit",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output merged h5ad path, e.g. merged_train_rat.h5ad",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_species = [s.strip() for s in args.train_species.split(",") if s.strip()]
    out_path = Path(args.out)

    if not train_species:
        raise ValueError("--train-species must contain at least one species.")

    adatas = []
    for sp in train_species:
        f = data_root / f"{sp}_control_ifn.h5ad"
        if not f.exists():
            raise FileNotFoundError(f"Expected file not found: {f}")
        a = ad.read_h5ad(f)
        a.obs["species"] = sp
        adatas.append(a)

    # var (gene) and, gene 
    merged = ad.concat(adatas, join="inner", label="batch", keys=train_species)
    merged.obs_names_make_unique()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(out_path)

    n_cells = merged.n_obs
    n_genes = merged.n_vars
    print(f"Written merged train h5ad: {out_path}  (cells={n_cells}, genes={n_genes}, species={train_species})")


if __name__ == "__main__":
    main()
