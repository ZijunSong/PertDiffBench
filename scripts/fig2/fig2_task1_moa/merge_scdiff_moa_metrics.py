#!/usr/bin/env python3
""" and fig2_task1 scDiff in MOA subdirunder scdiff/metrics/metrics_*_test.csv as CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def collect_split(
    task_root: Path, split: str
) -> tuple[list[str], list[list[str]]]:
    """task_root as task1_unseenMOA/same or .../diff, underas <MOA>/scdiff/metrics/*.csv."""
    paths = sorted(
        task_root.glob("*/scdiff/metrics/metrics_*_test.csv"),
        key=lambda p: p.parent.parent.parent.name,
    )
    out_rows: list[list[str]] = []
    base_header: list[str] | None = None
    for csv_path in paths:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            data = next(reader, None)
            if data is None:
                continue
        if base_header is None:
            base_header = header
        elif header != base_header:
            raise ValueError(
                f"header mismatch: {csv_path.name} and filecolsname , check headers before merge."
            )
        ds = csv_path.parent.parent.parent.name
        out_rows.append([split, ds] + data)

    if base_header is None:
        return [], []
    out_header = ["split", "Dataset"] + base_header
    return out_header, out_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--same-root",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/same"
        ),
        help="unseen_same_moa: with MOA subdir path",
    )
    p.add_argument(
        "--diff-root",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/diff"
        ),
        help="unseen_diff_moa: with MOA subdir path",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/scdiff_moa_combined_metrics.csv"
        ),
        help="output andafter CSV path",
    )
    p.add_argument(
        "--same-only",
        action="store_true",
        help=" and same, contain diff",
    )
    args = p.parse_args()

    same_header, same_rows = collect_split(args.same_root, "unseen_same_moa")
    if not same_rows:
        raise SystemExit(f" in {args.same_root}/*/scdiff/metrics/ found metrics_*_test.csv")

    if args.same_only:
        all_rows = [same_header] + same_rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(all_rows)
        print(f" {args.output} ( {len(same_rows)} , only same)")
        return

    diff_header, diff_rows = collect_split(args.diff_root, "unseen_diff_moa")
    if not diff_rows:
        raise SystemExit(f" in {args.diff_root}/*/scdiff/metrics/ found metrics_*_test.csv")

    if same_header != diff_header:
        raise SystemExit("same and diff header mismatch, cannot merge.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [same_header] + same_rows + diff_rows
    with args.output.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_rows)

    print(
        f" {args.output} ( {len(all_rows) - 1} data: "
        f"same {len(same_rows)} + diff {len(diff_rows)}）"
    )


if __name__ == "__main__":
    main()
