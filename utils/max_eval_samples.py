"""Compute maximum paired evaluation sample counts from AnnData files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


def _read_obs_column(h5ad_path: PathLike, column: str) -> np.ndarray:
    import scanpy as sc

    adata = sc.read_h5ad(str(h5ad_path), backed="r")
    if column not in adata.obs.columns:
        raise KeyError(f"Column '{column}' not found in {h5ad_path}")
    values = np.asarray(adata.obs[column].astype(str))
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return values


def max_paired_status_samples(
    h5ad_path: PathLike,
    status_col: str = "perturbation_status",
    control_value: str = "Control",
    stimulated_value: str = "IFN",
) -> int:
    """Max paired cells: min(control count, stimulated count) in one h5ad."""
    values = _read_obs_column(h5ad_path, status_col)
    n_ctrl = int(np.sum(values == control_value))
    n_stim = int(np.sum(values == stimulated_value))
    if n_ctrl < 1 or n_stim < 1:
        raise ValueError(
            f"{h5ad_path}: need both '{control_value}' and '{stimulated_value}' "
            f"(got control={n_ctrl}, stimulated={n_stim})"
        )
    return min(n_ctrl, n_stim)


def max_multipert_eval_samples(
    h5ad_path: PathLike,
    status_col: str = "perturbation_status",
    control_value: str = "Control",
    exclude_values: Optional[Sequence[str]] = None,
) -> int:
    """Max n_samples for multi-perturbation eval: min(control, smallest non-control group)."""
    values = _read_obs_column(h5ad_path, status_col)
    exclude = set(exclude_values or [])
    exclude.add(control_value)
    n_ctrl = int(np.sum(values == control_value))
    pert_counts = []
    for val in np.unique(values):
        if val in exclude:
            continue
        pert_counts.append(int(np.sum(values == val)))
    if n_ctrl < 1 or not pert_counts:
        raise ValueError(f"{h5ad_path}: insufficient control/perturbation groups")
    return min(n_ctrl, min(pert_counts))


def max_timepoint_eval_samples(
    h5ad_path: PathLike,
    time_col: str = "treatment_time",
    exclude_times: Optional[Sequence[str]] = None,
) -> int:
    """Fig4-style: min cells across non-excluded time points."""
    values = _read_obs_column(h5ad_path, time_col)
    exclude = set(exclude_times or ["0h"])
    counts = [int(np.sum(values == t)) for t in np.unique(values) if t not in exclude]
    if not counts:
        raise ValueError(f"{h5ad_path}: no evaluable time points in '{time_col}'")
    return min(counts)


def resolve_eval_n_samples(
    h5ad_path: PathLike,
    requested: Optional[int] = None,
    *,
    mode: str = "paired_ifn",
    status_col: str = "perturbation_status",
    control_value: str = "Control",
    stimulated_value: str = "IFN",
    time_col: str = "treatment_time",
) -> int:
    """Return min(requested, max_available) when requested>0; else max_available."""
    if mode == "paired_ifn":
        available = max_paired_status_samples(
            h5ad_path, status_col, control_value, stimulated_value
        )
    elif mode == "multi_pert":
        available = max_multipert_eval_samples(h5ad_path, status_col, control_value)
    elif mode == "timepoint":
        available = max_timepoint_eval_samples(h5ad_path, time_col)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if requested is None or requested <= 0:
        return available
    return min(int(requested), available)
