#!/usr/bin/env python3
"""
 fig2_task2_extend under .sh results CSV as CSV file.

CSV ( forrepo root): 
- scgen:           samples/fig2/task2_extend_scgen/scgen/metrics_all.csv
- ddpm:            samples/fig2/task2_extend_scgen/pretrain_CD4T/scrna_ddpm_scrna/metrics_all.csv
- ddpm_mlp:        samples/fig2/task2_extend_scgen/pretrain_CD4T/mlp_ddpm_mlp/metrics_all.csv
- scdiffusion:     samples/fig2/task2_extend_scgen/scDiffusion/metrics_all.csv
- squidiff:        samples/fig2/task2_extend_scgen/squidiff/metrics_all.csv
- scdiff (B/NK):   samples/fig2/task2_extend_scgen/scdiff/task2_B/metrics_task2_B.csv
                   samples/fig2/task2_extend_scgen/scdiff/task2_NK/metrics_task2_NK.csv

for" " metrics_all.csv ( logicas ), keepeachfile 2 
 (forshould B, NK test cell type), scGen results.
"""

import csv
import os
from pathlib import Path

# repo root: scripts/fig2/fig2_task2_extend/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# CSV path ( for REPO_ROOT)
# : (path, whether Dataset cols, canfrompath dataset name, whether take only after N )
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
    """ CSV, return (headers, rows). Dataset cols on.
    append_style=True when keep after NUM_TARGET_CELL_TYPES ( results).
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not rows:
        return header, []

    # CSV: keep 2 (B, NK)
    if append_style and len(rows) > NUM_TARGET_CELL_TYPES:
        rows = rows[-NUM_TARGET_CELL_TYPES:]

    # as (Dataset, Method, ...) header
    if has_dataset:
        if header[0] != "Dataset":
            return header, rows
        return header, rows

    # scdiff : Method infirstcols, Dataset
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
            print(f"skip ( exist): {path}")
            continue
        seen_paths_ok.append(rel_path)
        header, rows = read_csv_rows(path, has_dataset, default_dataset, append_style)
        if all_headers is None:
            all_headers = header
        else:
            # checkcolsname ( cols when colsnamealign, here requires )
            if header != all_headers:
                # Dataset in cols, can 
                if header == ["Dataset"] + all_headers[1:]:
                    pass
                elif ["Dataset"] + header[1:] == all_headers:
                    pass
                else:
                    print(f" : colsnameand file , filecolsname and: {rel_path}")
        for row in rows:
            if len(row) != len(all_headers):
                # or to header
                if len(row) < len(all_headers):
                    row = row + [""] * (len(all_headers) - len(row))
                else:
                    row = row[: len(all_headers)]
            all_rows.append(row)

    if not all_headers or not all_rows:
        print(" found can anddata , CSV fileexist empty.")
        return

    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(all_rows)

    print(f" and {len(seen_paths_ok)} CSV, {len(all_rows)} -> {OUTPUT_CSV}")
    for p in seen_paths_ok:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
