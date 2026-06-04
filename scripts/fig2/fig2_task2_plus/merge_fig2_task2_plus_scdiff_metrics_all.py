#!/usr/bin/env python3
"""
Merge per-folder metrics_*.csv under fig2_task2_plus/scdiff into one metrics_all.csv
(same layout as samples/fig2/fig2_task2_plus/scDiffusion/metrics_all.csv: Dataset + Method + metrics...).

Usage:
  python merge_fig2_task2_plus_scdiff_metrics_all.py
  python merge_fig2_task2_plus_scdiff_metrics_all.py --root /path/to/scdiff --output /path/to/metrics_all.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Same holdout / control order as fig2_task2_plus_scdiff.sh (for stable row ordering).
HOLDOUT_ORDER = ["B", "CD4T", "CD8T", "CD14+Mono", "Dendritic", "FCGR3A+Mono", "NK"]
SLUG_ORDER = ["p0", "p0.25", "p0.5"]


def _sort_key(dataset: str) -> tuple[int, int, str]:
    """Sort key: (holdout_index, slug_index, dataset) for tie-break."""
    for si, slug in enumerate(SLUG_ORDER):
        suffix = "_" + slug
        if dataset.endswith(suffix):
            ht = dataset[: -len(suffix)]
            try:
                hi = HOLDOUT_ORDER.index(ht)
            except ValueError:
                hi = len(HOLDOUT_ORDER)
            return (hi, si, dataset)
    return (len(HOLDOUT_ORDER), len(SLUG_ORDER), dataset)


def find_metrics_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        tag = child.name
        cand = child / f"metrics_{tag}.csv"
        if cand.is_file():
            out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    default_root = repo / "samples/fig2/fig2_task2_plus/scdiff"
    ap.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Directory containing <DatasetTag>/metrics_<DatasetTag>.csv (default: {default_root})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <root>/metrics_all.csv)",
    )
    args = ap.parse_args()
    root: Path = args.root
    out_path: Path = args.output if args.output is not None else root / "metrics_all.csv"

    if not root.is_dir():
        raise SystemExit(f"root is not a directory: {root}")

    files = find_metrics_files(root)
    if not files:
        raise SystemExit(f"No metrics_<tag>.csv found under subfolders of {root}")

    rows_out: list[dict[str, str]] = []
    header_merged: list[str] | None = None

    for path in files:
        dataset = path.parent.name
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise SystemExit(f"Empty or invalid CSV: {path}")
            if reader.fieldnames[0] != "Method":
                raise SystemExit(
                    f"Expected first column 'Method' in {path}, got {reader.fieldnames[0]!r}"
                )
            if header_merged is None:
                header_merged = ["Dataset"] + list(reader.fieldnames)
            elif list(reader.fieldnames) != header_merged[1:]:
                raise SystemExit(
                    f"Header mismatch:\n  {path}\nvs expected columns after Dataset: {header_merged[1:]}"
                )
            data_rows = list(reader)
            if len(data_rows) != 1:
                raise SystemExit(
                    f"Expected exactly 1 data row in {path}, got {len(data_rows)}"
                )
            row = data_rows[0]
            rows_out.append({"Dataset": dataset, **row})

    rows_out.sort(key=lambda r: _sort_key(r["Dataset"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        assert header_merged is not None
        w = csv.DictWriter(f, fieldnames=header_merged, lineterminator="\n")
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
