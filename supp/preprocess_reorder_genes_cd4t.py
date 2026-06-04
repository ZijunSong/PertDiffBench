#!/usr/bin/env python3
"""Generate CD4T HVG-1000 h5ad with shuffled or cluster-reordered gene columns.

Train and valid share the same gene order. Does NOT aggregate cluster means;
only reorders columns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _align_valid_to_train_genes(
    train: ad.AnnData,
    valid: ad.AnnData,
    cell_type: str,
) -> ad.AnnData:
    """Slice valid to train gene list (same names, same order).

    Released HVG files compute top genes separately on train/valid splits, so
    column names can differ even when both have n_genes columns. For reordering
    experiments we must share one gene ordering; train defines the canonical set.
    """
    genes = list(train.var_names)
    missing = [g for g in genes if g not in valid.var_names]
    if missing:
        raise ValueError(
            f"Valid data missing {len(missing)} train genes for {cell_type}. "
            f"Examples: {missing[:5]}. Use full-split h5ad as --valid-src."
        )
    valid_aligned = valid[:, genes].copy()
    if list(valid_aligned.var_names) != genes:
        raise RuntimeError("Failed to align valid genes to train order.")
    return valid_aligned


def _load_pair(
    src_dir: Path,
    valid_src: Path,
    cell_type: str,
    n_genes: int,
) -> tuple[ad.AnnData, ad.AnnData]:
    train_path = src_dir / f"{cell_type}_train_HVG_{n_genes}.h5ad"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train h5ad: {train_path}")

    train = ad.read_h5ad(train_path)
    valid_path = valid_src / f"task1_valid_{cell_type}_exp.h5ad"
    if not valid_path.exists():
        # Fallback: try HVG valid and realign if gene sets match
        hvg_valid_path = src_dir / f"{cell_type}_valid_HVG_{n_genes}.h5ad"
        if not hvg_valid_path.exists():
            raise FileNotFoundError(
                f"Missing valid h5ad: {valid_path} or {hvg_valid_path}"
            )
        valid_raw = ad.read_h5ad(hvg_valid_path)
    else:
        valid_raw = ad.read_h5ad(valid_path)

    if list(train.var_names) == list(valid_raw.var_names):
        valid = valid_raw
    else:
        print(
            f"Note: realigning valid genes to train HVG list "
            f"(train/valid HVG files were selected independently)."
        )
        valid = _align_valid_to_train_genes(train, valid_raw, cell_type)
    return train, valid


def _reorder(adata: ad.AnnData, order: list[str]) -> ad.AnnData:
    return adata[:, order].copy()


def shuffle_order(gene_names: list[str], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(gene_names))
    return [gene_names[i] for i in perm]


def cluster_order(train: ad.AnnData, method: str = "average") -> list[str]:
    """Hierarchical clustering on gene-gene Pearson correlation (train only)."""
    adata = train.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    # genes x genes correlation
    corr = np.corrcoef(x.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - corr
    dist = np.clip(dist, 0.0, None)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method=method)
    leaf_idx = leaves_list(z)
    genes = list(train.var_names)
    return [genes[i] for i in leaf_idx]


def save_outputs(
    train: ad.AnnData,
    valid: ad.AnnData,
    order: list[str],
    out_dir: Path,
    cell_type: str,
    n_genes: int,
    tag: str,
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = _reorder(train, order)
    valid_out = _reorder(valid, order)
    train_out.write(out_dir / f"{cell_type}_train_HVG_{n_genes}.h5ad")
    valid_out.write(out_dir / f"{cell_type}_valid_HVG_{n_genes}.h5ad")
    order_path = out_dir / f"{cell_type}_gene_order_{tag}.json"
    with open(order_path, "w", encoding="utf-8") as f:
        json.dump({"mode": tag, **meta, "gene_order": order}, f, indent=2)
    print(f"Wrote {train_out.shape} train / {valid_out.shape} valid -> {out_dir}")
    print(f"Gene order metadata -> {order_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("/data/ppnm/data/PertDiffBench/data/highly_variable_gene_gradient"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data/ppnm/data/PertDiffBench/data/gene_order_exp"),
    )
    parser.add_argument(
        "--valid-src",
        type=Path,
        default=Path("/data/ppnm/data/PertDiffBench/data/fig1_task1"),
        help="Full valid split h5ad dir (task1_valid_{cell}_exp.h5ad); used to align genes to train.",
    )
    parser.add_argument("--cell-type", default="CD4T")
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["shuffle", "cluster", "both"],
        default="both",
    )
    args = parser.parse_args()

    train, valid = _load_pair(args.src_dir, args.valid_src, args.cell_type, args.n_genes)
    genes = list(train.var_names)

    if args.mode in ("shuffle", "both"):
        order = shuffle_order(genes, args.seed)
        save_outputs(
            train,
            valid,
            order,
            args.out_root / "shuffle",
            args.cell_type,
            args.n_genes,
            "shuffle",
            {"seed": args.seed, "n_genes": args.n_genes},
        )

    if args.mode in ("cluster", "both"):
        order = cluster_order(train)
        save_outputs(
            train,
            valid,
            order,
            args.out_root / "cluster",
            args.cell_type,
            args.n_genes,
            "cluster",
            {"linkage": "average", "n_genes": args.n_genes},
        )


if __name__ == "__main__":
    main()
