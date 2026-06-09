"""Parse evaluation stdout and aggregate metrics across runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

METRIC_PATTERNS = {
    "PDS": [
        r"Perturbation Discrimination Score \(PDS\):\s*([-\d.eE+]+|nan|N/A)",
        r"^PDS:\s*([-\d.eE+]+|nan|N/A)",
    ],
    "MAE": [
        r"Mean Absolute Error \(MAE\):\s*([-\d.eE+]+|nan|N/A)",
        r"^MAE:\s*([-\d.eE+]+|nan|N/A)",
    ],
    "DES": [
        r"Differential Expression Score \(DES\):\s*([-\d.eE+]+|nan|N/A)",
        r"^DES:\s*([-\d.eE+]+|nan|N/A)",
    ],
    "E-Distance": [
        r"E-Distance:\s*([-\d.eE+]+|nan|N/A)",
        r"^E-distance:\s*([-\d.eE+]+|nan|N/A)",
    ],
    "MMD": [
        r"Maximum Mean Discrepancy \(MMD\):\s*([-\d.eE+]+|nan|N/A)",
        r"^MMD:\s*([-\d.eE+]+|nan|N/A)",
    ],
    "R2": [
        r"R-squared \(R2\):\s*([-\d.eE+]+|nan|N/A)",
        r"^R2:\s*([-\d.eE+]+|nan|N/A)",
    ],
    "Pearson (all genes)": [
        r"Pearson \(all genes\):\s*([-\d.eE+]+|nan|N/A)",
        r"^Pearson\(all genes\):\s*([-\d.eE+]+|nan|N/A)",
    ],
    "Pearson Delta (all genes)": [
        r"Pearson Delta \(all genes\):\s*([-\d.eE+]+|nan|N/A)",
        r"^Pearson Delta\(all genes\):\s*([-\d.eE+]+|nan|N/A)",
    ],
    "Pearson Delta (top 20 DE genes)": [
        r"Pearson Delta \(top 20 DE genes\):\s*([-\d.eE+]+|nan|N/A)",
        r"^Pearson Delta\(top 20 DE genes\):\s*([-\d.eE+]+|nan|N/A)",
    ],
    "Pearson Delta (top 50 DE genes)": [
        r"Pearson Delta \(top 50 DE genes\):\s*([-\d.eE+]+|nan|N/A)",
        r"^Pearson Delta\(top 50 DE genes\):\s*([-\d.eE+]+|nan|N/A)",
    ],
    "Pearson Delta (top 100 DE genes)": [
        r"Pearson Delta \(top 100 DE genes\):\s*([-\d.eE+]+|nan|N/A)",
        r"^Pearson Delta\(top 100 DE genes\):\s*([-\d.eE+]+|nan|N/A)",
    ],
}

METRIC_NAMES = list(METRIC_PATTERNS.keys())


def _to_float(value: str) -> float:
    value = value.strip()
    if value in {"nan", "N/A", "N/A (No data)"}:
        return float("nan")
    return float(value)


def parse_metrics_from_output(output: str) -> Dict[str, float]:
    """Extract the 11 benchmark metrics from eval script stdout."""
    metrics: Dict[str, float] = {}
    for name, patterns in METRIC_PATTERNS.items():
        val = float("nan")
        for pattern in patterns:
            match = re.search(pattern, output, re.MULTILINE)
            if match:
                val = _to_float(match.group(1))
                break
        metrics[name] = val
    return metrics


@dataclass
class RunMetrics:
    method: str
    run_index: int
    values: Dict[str, float] = field(default_factory=dict)


def aggregate_runs(method: str, run_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """Build a summary DataFrame with mean±std and per-run columns."""
    if not run_metrics:
        return pd.DataFrame()

    summary: Dict[str, object] = {"Method": method}

    for metric in METRIC_NAMES:
        vals = [m.get(metric, float("nan")) for m in run_metrics]
        series = pd.Series(vals, dtype=float)
        mean = series.mean()
        std = series.std(ddof=1) if len(series) > 1 else 0.0
        summary[f"{metric} (mean±std)"] = f"{mean:.4f}±{std:.4f}"
        for i, v in enumerate(vals, start=1):
            summary[f"Run{i} {metric}"] = v

    return pd.DataFrame([summary])


def save_metrics_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
