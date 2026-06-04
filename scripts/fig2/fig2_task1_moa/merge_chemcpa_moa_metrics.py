#!/usr/bin/env python3
"""合并 fig2_task1 ChemCPA 在 unseen_same_moa / unseen_diff_moa 下各 MOA 的 metrics_*.csv。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_metrics_rows(metrics_dir: Path, split: str) -> list[list[str]]:
    parsed: list[tuple[list[str], list[str], str]] = []
    for csv_path in sorted(metrics_dir.glob("metrics_*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                continue
            data = next(reader, None)
            if data is None:
                continue
            parsed.append((header, data, csv_path.name))
    if not parsed:
        return []

    base_header = parsed[0][0]
    for h, _, name in parsed:
        if h != base_header:
            raise ValueError(
                f"表头不一致: {name} 与首个文件列名不同，请检查后再合并。"
            )

    out_header = ["split"] + base_header
    out_rows = [[split] + d for _, d, _ in parsed]
    return [out_header] + out_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--same-dir",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/same/chemcpa/metrics"
        ),
        help="unseen_same_moa 的 metrics 目录",
    )
    p.add_argument(
        "--diff-dir",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/diff/chemcpa/metrics"
        ),
        help="unseen_diff_moa 的 metrics 目录",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/chemcpa_moa_combined_metrics.csv"
        ),
        help="输出合并后的 CSV 路径",
    )
    args = p.parse_args()

    same_block = read_metrics_rows(args.same_dir, "unseen_same_moa")
    diff_block = read_metrics_rows(args.diff_dir, "unseen_diff_moa")

    if len(same_block) <= 1:
        raise SystemExit(f"未在 {args.same_dir} 读到任何 metrics_*.csv 数据行")
    if len(diff_block) <= 1:
        raise SystemExit(f"未在 {args.diff_dir} 读到任何 metrics_*.csv 数据行")

    same_header, *same_rows = same_block
    diff_header, *diff_rows = diff_block
    if same_header != diff_header:
        raise SystemExit("same 与 diff 两侧表头不一致，无法合并。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [same_header] + same_rows + diff_rows
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(all_rows)

    print(
        f"已写入 {args.output}（共 {len(all_rows) - 1} 行数据："
        f"same {len(same_rows)} + diff {len(diff_rows)}）"
    )


if __name__ == "__main__":
    main()
