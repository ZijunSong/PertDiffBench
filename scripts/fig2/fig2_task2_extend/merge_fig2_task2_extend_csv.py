#!/usr/bin/env python3
"""
汇总 fig2_task2_extend 下各 .sh 脚本生成的实验结果 CSV 为单个 CSV 文件。

CSV 来源（均相对于项目根目录）：
- scgen:           samples/fig2/task2_extend_scgen/scgen/metrics_all.csv
- ddpm:            samples/fig2/task2_extend_scgen/pretrain_CD4T/scrna_ddpm_scrna/metrics_all.csv
- ddpm_mlp:        samples/fig2/task2_extend_scgen/pretrain_CD4T/mlp_ddpm_mlp/metrics_all.csv
- scdiffusion:     samples/fig2/task2_extend_scgen/scDiffusion/metrics_all.csv
- squidiff:        samples/fig2/task2_extend_scgen/squidiff/metrics_all.csv
- scdiff (B/NK):   samples/fig2/task2_extend_scgen/scdiff/task2_B/metrics_task2_B.csv
                   samples/fig2/task2_extend_scgen/scdiff/task2_NK/metrics_task2_NK.csv

对“续写型”的 metrics_all.csv（脚本逻辑为追加而非覆盖），只保留每个文件中最新 2 行
（对应 B、NK 两个 test cell type），避免重复的 scGen 等结果。
"""

import csv
import os
from pathlib import Path

# 项目根目录：脚本位于 scripts/fig2/fig2_task2_extend/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# 各方法生成的 CSV 路径（相对 REPO_ROOT）
# 格式: (路径, 是否已有 Dataset 列, 若无可从路径推断的 dataset 名, 是否续写型只取最后 N 行)
NUM_TARGET_CELL_TYPES = 2  # B, NK

CSV_SOURCES = [
    ("samples/fig2/task2_extend_scgen/scgen/metrics_all.csv", True, None, True),
    ("samples/fig2/task2_extend_scgen/pretrain_CD4T/scrna_ddpm_scrna/metrics_all.csv", True, None, True),
    ("samples/fig2/task2_extend_scgen/pretrain_CD4T/mlp_ddpm_mlp/metrics_all.csv", True, None, True),
    ("samples/fig2/task2_extend_scgen/scDiffusion/metrics_all.csv", True, None, True),
    ("samples/fig2/task2_extend_scgen/squidiff/metrics_all.csv", True, None, True),
    ("samples/fig2/task2_extend_scgen/scdiff/task2_B/metrics_task2_B.csv", False, "B", False),
    ("samples/fig2/task2_extend_scgen/scdiff/task2_NK/metrics_task2_NK.csv", False, "NK", False),
]

OUTPUT_CSV = SCRIPT_DIR / "fig2_task2_extend_metrics_merged.csv"


def read_csv_rows(
    path: Path,
    has_dataset: bool,
    default_dataset: str | None,
    append_style: bool,
):
    """读取单个 CSV，返回 (headers, rows)。若缺少 Dataset 列则补上。
    append_style=True 时只保留最后 NUM_TARGET_CELL_TYPES 行（最新两次实验结果）。
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        return header, []

    # 续写型 CSV：只保留最新 2 行（B、NK）
    if append_style and len(rows) > NUM_TARGET_CELL_TYPES:
        rows = rows[-NUM_TARGET_CELL_TYPES:]

    # 统一为 (Dataset, Method, ...) 表头
    if has_dataset:
        if header[0] != "Dataset":
            return header, rows
        return header, rows

    # scdiff 格式: Method 在第一列，无 Dataset
    if header[0] == "Method":
        new_header = ["Dataset"] + header
        new_rows = [[default_dataset] + row for row in rows]
        return new_header, new_rows

    return header, rows


def main():
    all_headers = None
    all_rows = []
    seen_paths_ok = []

    for item in CSV_SOURCES:
        if len(item) == 4:
            rel_path, has_dataset, default_dataset, append_style = item
        else:
            rel_path, has_dataset, default_dataset = item[0], item[1], item[2]
            append_style = "metrics_all.csv" in rel_path
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"跳过（不存在）: {path}")
            continue
        seen_paths_ok.append(rel_path)
        header, rows = read_csv_rows(path, has_dataset, default_dataset, append_style)
        if all_headers is None:
            all_headers = header
        else:
            # 检查列名一致（允许列顺序不同时按列名对齐，这里简单要求一致）
            if header != all_headers:
                # 若只是多了 Dataset 在首列，可接受
                if header == ["Dataset"] + all_headers[1:]:
                    pass
                elif ["Dataset"] + header[1:] == all_headers:
                    pass
                else:
                    print(f"警告: 列名与首文件不一致，将按首文件列名合并: {rel_path}")
        for row in rows:
            if len(row) != len(all_headers):
                # 补齐或截断以匹配表头
                if len(row) < len(all_headers):
                    row = row + [""] * (len(all_headers) - len(row))
                else:
                    row = row[: len(all_headers)]
            all_rows.append(row)

    if not all_headers or not all_rows:
        print("未找到任何可合并的数据行，请确认 CSV 文件存在且非空。")
        return

    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_rows)

    print(f"已合并 {len(seen_paths_ok)} 个 CSV，共 {len(all_rows)} 行 -> {OUTPUT_CSV}")
    for p in seen_paths_ok:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
