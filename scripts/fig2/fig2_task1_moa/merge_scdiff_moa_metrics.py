#!/usr/bin/env python3
"""合并 fig2_task1 scDiff 在各 MOA 子目录下 scdiff/metrics/metrics_*_test.csv 为单一 CSV。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def collect_split(
    task_root: Path, split: str
) -> tuple[list[str], list[list[str]]]:
    """task_root 为 task1_unseenMOA/same 或 .../diff，其下为 <MOA>/scdiff/metrics/*.csv。"""
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
                f"表头不一致: {csv_path.name} 与首个文件列名不同，请检查后再合并。"
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
        help="unseen_same_moa：含各 MOA 子目录的根路径",
    )
    p.add_argument(
        "--diff-root",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/diff"
        ),
        help="unseen_diff_moa：含各 MOA 子目录的根路径",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "/data/ppnm/data/PertDiffBench/samples/fig2/task1_unseenMOA/scdiff_moa_combined_metrics.csv"
        ),
        help="输出合并后的 CSV 路径",
    )
    p.add_argument(
        "--same-only",
        action="store_true",
        help="只合并 same，不包含 diff",
    )
    args = p.parse_args()

    same_header, same_rows = collect_split(args.same_root, "unseen_same_moa")
    if not same_rows:
        raise SystemExit(f"未在 {args.same_root}/*/scdiff/metrics/ 找到 metrics_*_test.csv")

    if args.same_only:
        all_rows = [same_header] + same_rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(all_rows)
        print(f"已写入 {args.output}（共 {len(same_rows)} 行，仅 same）")
        return

    diff_header, diff_rows = collect_split(args.diff_root, "unseen_diff_moa")
    if not diff_rows:
        raise SystemExit(f"未在 {args.diff_root}/*/scdiff/metrics/ 找到 metrics_*_test.csv")

    if same_header != diff_header:
        raise SystemExit("same 与 diff 两侧表头不一致，无法合并。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [same_header] + same_rows + diff_rows
    with args.output.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(all_rows)

    print(
        f"已写入 {args.output}（共 {len(all_rows) - 1} 行数据："
        f"same {len(same_rows)} + diff {len(diff_rows)}）"
    )


if __name__ == "__main__":
    main()
