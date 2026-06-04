#!/usr/bin/env python3
"""
Aggregate multi-MOA experiment CSV results into one aggregate CSV.
Each per-MOA CSV contributes its latest (last) row.
"""

import argparse
import os
import glob
import pandas as pd
from pathlib import Path


def get_latest_row_from_csv(csv_path):
    """
    Read the last non-empty row from a CSV file.
    
    Args:
        csv_path: CSVfile path
        
    Returns:
        Last row as a string, or None if file is empty/missing
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file not found: {csv_path}")
        return None
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            print(f"[WARN] CSV file is empty: {csv_path}")
            return None
        
        # Return last row
        return lines[-1]
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_path}: {e}")
        return None


def generate_csv_header(num_runs=None):
    """
    generate CSVfile header.
    
    Args:
        num_runs: number of runs; if None, only mean±std columns (no per-run columns)
        
    Returns:
        Header string
    """
    # 11 metric names (script order)
    metric_names = [
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
        "Pearson Delta (top 100 DE genes) (mean±std)"
    ]
    
    # Per-run metric names (without mean±std)
    metric_names_raw = [
        "PDS",
        "MAE",
        "DES",
        "E-Distance",
        "MMD",
        "R2",
        "Pearson (all genes)",
        "Pearson Delta (all genes)",
        "Pearson Delta (top 20 DE genes)",
        "Pearson Delta (top 50 DE genes)",
        "Pearson Delta (top 100 DE genes)"
    ]
    
    # buildheader
    header = ["Dataset", "Method"]
    
    # add mean±std columns 
    header.extend(metric_names)
    
    # If num_runs set, add per-run metric columns
    if num_runs is not None and num_runs > 0:
        for run_num in range(1, num_runs + 1):
            for metric in metric_names_raw:
                header.append(f"Run{run_num} {metric}")
    
    return ",".join(header)


def aggregate_metrics_csvs(samples_root, output_csv_path, pattern="scDiffusion_3000"):
    """
    Aggregate all MOA metrics CSVs into one file.
    
    Args:
        samples_root: e.g. samples/fig2/task1_unseen_moa_same
        output_csv_path: outputaggregateCSVpath to file
        pattern: subdirectory glob, e.g. "scDiffusion_3000"
        
    Returns:
        aggregateafter  CSVfileabsolute path
    """
    samples_root = Path(samples_root).resolve()
    output_csv_path = Path(output_csv_path).resolve()
    
    # Find all matching CSV files
    csv_pattern = str(samples_root / "*" / pattern / "metrics" / "metrics_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"[ERROR] No CSV files found matching pattern: {csv_pattern}")
        return None
    
    print(f"Found {len(csv_files)} CSV files to aggregate")
    
    # Collect all data rows
    aggregated_rows = []
    moa_names = []
    num_runs = None
    row_has_dataset_col = True  # whether rows already include Dataset column
    
    for csv_file in sorted(csv_files):
        # Extract MOA name from path
        # Path: .../samples_root/MOA_NAME/scDiffusion_3000/metrics/metrics_MOA_NAME.csv
        parts = Path(csv_file).parts
        try:
            # Locate samples_root in path
            root_idx = None
            for i, part in enumerate(parts):
                if samples_root.name in part or str(samples_root) in '/'.join(parts[:i+1]):
                    root_idx = i
                    break
            
            if root_idx is not None and root_idx + 1 < len(parts):
                moa_name = parts[root_idx + 1]
            else:
                # Fallback: parse from filename
                filename = Path(csv_file).stem
                moa_name = filename.replace('metrics_', '').replace('_test', '')
            
            latest_row = get_latest_row_from_csv(csv_file)
            if latest_row:
                # Infer column count and num_runs from first row
                if num_runs is None:
                    num_cols = len(latest_row.split(','))
                    # total cols = 2 (Dataset+Method) + 11 (mean±std) + num_runs*11
                    # if each rowsonly  has  Method + 11 + num_runs*11 ( no  Dataset), then  num_cols = 1+11+11*num_runs
                    if num_cols >= 13 and (num_cols - 1 - 11) % 11 == 0:
                        num_runs = (num_cols - 1 - 11) // 11  #  no  Dataset  columns 
                        row_has_dataset_col = False
                        print(f"Detected format: no Dataset column, {num_runs} runs (total columns: {num_cols})")
                    elif num_cols == 13:
                        num_runs = 0
                        row_has_dataset_col = False
                        print(f"Detected format: mean±std only (total columns: {num_cols})")
                    else:
                        num_runs = (num_cols - 2 - 11) // 11
                        if num_runs <= 0:
                            print(f"[WARN] Could not determine number of runs from column count: {num_cols}")
                            num_runs = 0
                        else:
                            print(f"Detected {num_runs} runs per MOA (total columns: {num_cols})")
                
                row_to_append = latest_row
                if not row_has_dataset_col:
                    row_to_append = moa_name + "," + latest_row
                aggregated_rows.append(row_to_append)
                moa_names.append(moa_name)
                print(f"  ✓ {moa_name}: {csv_file}")
            else:
                print(f"  ✗ {moa_name}: No valid data found")
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_file}: {e}")
    
    if not aggregated_rows:
        print("[ERROR] No valid data rows found to aggregate")
        return None
    
    # ensureoutputdir exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # wroteaggregateCSVfile
    try:
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            # wroteheader
            # num_runs==0: mean±std only; None: not detected
            if num_runs is not None:
                header = generate_csv_header(num_runs if num_runs > 0 else None)
                f.write(header + '\n')
            else:
                print("[WARN] Could not generate header, writing without header")
            
            # Write all data rows
            for row in aggregated_rows:
                f.write(row + '\n')
        
        print(f"\n[SUCCESS] Aggregated {len(aggregated_rows)} MOA results to: {output_csv_path}")
        print(f"MOAs included: {', '.join(moa_names)}")
        return str(output_csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to write aggregated CSV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-MOA experiment CSVs into one aggregate CSV"
    )
    parser.add_argument(
        "--samples-root",
        type=str,
        required=True,
        help="Samples root path, e.g. samples/fig2/task1_unseen_moa_same"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="outputaggregateCSVpath to file"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="scDiffusion_3000",
        help="Subdirectory pattern to match (default: scDiffusion_3000)"
    )
    
    args = parser.parse_args()
    
    result_path = aggregate_metrics_csvs(
        args.samples_root,
        args.output_csv,
        args.pattern
    )
    
    if result_path:
        print(f"\nAggregated CSV absolute path: {os.path.abspath(result_path)}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
