#!/usr/bin/env python3
"""Merge fig2 task1 MOA metrics for all 6 log scripts into one CSV.

Covers:
  - DDPM same/diff
  - DDPM+MLP same/diff
  - ChemCPA same/diff
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

SAMPLES_ROOT = Path("/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA")

METRIC_MEAN_STD = [
    "PDS (mean±std)",
    "MAE (mean±std)",
    "DES (mean±std)",
    "E-Distance (mean±std)",
    "MMD (mean±std)",
    "R2 (mean±std)",
    "Pearson (all genes) (mean±std)",
    "Pearson Delta (all genes) (mean±std)",
    "Pearson Delta (top 20 DE genes) (mean±std)",
    "Pearson Delta (top 50 DE genes) (mean±std)",
    "Pearson Delta (top 100 DE genes) (mean±std)",
]

RUN_METRICS = [
    "PDS", "MAE", "DES", "E-Distance", "MMD", "R2",
    "Pearson (all genes)", "Pearson Delta (all genes)",
    "Pearson Delta (top 20 DE genes)", "Pearson Delta (top 50 DE genes)",
    "Pearson Delta (top 100 DE genes)",
]


def _header_with_runs(num_runs: int = 3) -> list[str]:
    h = ["split", "dataset", "method"] + METRIC_MEAN_STD
    for r in range(1, num_runs + 1):
        for m in RUN_METRICS:
            h.append(f"Run{r} {m}")
    return h


def _read_csv_row(path: Path) -> tuple[list[str] | None, list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    if rows[0][0] in ("Dataset", "dataset", "split"):
        header, data = rows[0], rows[1] if len(rows) > 1 else None
    else:
        header, data = None, rows[-1]
    if data is None:
        return header, []
    return header, data


def _parse_chemcpa_row(header: list[str], data: list[str], split: str) -> dict[str, str]:
    idx = {name: i for i, name in enumerate(header)}
    out = {
        "split": split,
        "dataset": data[idx["Dataset"]],
        "method": data[idx["Method"]],
    }
    for col in METRIC_MEAN_STD:
        out[col] = data[idx[col]] if col in idx else ""
    for r in range(1, 4):
        for m in RUN_METRICS:
            key = f"Run{r} {m}"
            out[key] = data[idx[key]] if key in idx else ""
    return out


def _parse_ddpm_row(data: list[str], split: str, method: str) -> dict[str, str]:
    # Raw DDPM rows: dataset_test, method, PDS, MAE, ... (no header)
    dataset = data[0].removesuffix("_test") if data[0].endswith("_test") else data[0]
    out = {"split": split, "dataset": dataset, "method": method}
    for i, col in enumerate(METRIC_MEAN_STD, start=2):
        out[col] = data[i] if i < len(data) else ""
    for r in range(1, 4):
        for m in RUN_METRICS:
            out[f"Run{r} {m}"] = ""
    return out


def collect_rows(split: str, split_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for path in sorted(split_dir.glob("chemcpa/metrics/metrics_*.csv")):
        header, data = _read_csv_row(path)
        if not data:
            continue
        if header:
            rows.append(_parse_chemcpa_row(header, data, split))
        else:
            rows.append(_parse_ddpm_row(data, split, "ChemCPA"))

    for path in sorted(split_dir.glob("*/DDPM_3000/metrics/metrics_*.csv")):
        _, data = _read_csv_row(path)
        if data:
            method = data[1] if len(data) > 1 and "DDPM" in data[1] else "DDPM(3000)"
            rows.append(_parse_ddpm_row(data, split, method))

    for path in sorted(split_dir.glob("*/DDPM_MLP_3000/metrics/metrics_*.csv")):
        _, data = _read_csv_row(path)
        if data:
            method = data[1] if len(data) > 1 and "DDPM" in data[1] else "DDPM+MLP(3000)"
            rows.append(_parse_ddpm_row(data, split, method))

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples-root",
        type=Path,
        default=SAMPLES_ROOT,
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SAMPLES_ROOT / "fig2_task1_moa_all_methods_metrics.csv",
    )
    args = p.parse_args()

    split_map = {
        "same": "unseen_same_moa",
        "diff": "unseen_diff_moa",
    }
    all_rows: list[dict[str, str]] = []
    for subdir, split_name in split_map.items():
        split_path = args.samples_root / subdir
        if not split_path.is_dir():
            raise SystemExit(f"Missing directory: {split_path}")
        all_rows.extend(collect_rows(split_name, split_path))

    header = _header_with_runs(3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    methods = sorted({r["method"] for r in all_rows})
    splits = sorted({r["split"] for r in all_rows})
    print(f"Wrote {args.output}")
    print(f"  rows={len(all_rows)} | splits={splits} | methods={methods}")
    for split in splits:
        for method in methods:
            n = sum(1 for r in all_rows if r["split"] == split and r["method"] == method)
            if n:
                print(f"  {split} / {method}: {n} MOAs")


if __name__ == "__main__":
    main()
