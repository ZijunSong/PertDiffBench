#!/usr/bin/env python3
"""
汇总多个MOA实验结果的CSV文件到一个汇总CSV文件中。
每个单独的CSV文件取最后一行（最新的结果）。
"""

import argparse
import os
import glob
import pandas as pd
from pathlib import Path


def get_latest_row_from_csv(csv_path):
    """
    从CSV文件中读取最后一行（非空行）数据。
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        最后一行数据的字符串，如果文件为空或不存在则返回None
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
        
        # 返回最后一行
        return lines[-1]
    except Exception as e:
        print(f"[ERROR] Failed to read {csv_path}: {e}")
        return None


def generate_csv_header(num_runs=None):
    """
    生成CSV文件的表头。
    
    Args:
        num_runs: 运行的次数。如果为None，则只生成mean±std列（不包含单独的run列）
        
    Returns:
        表头字符串
    """
    # 11个指标的名称（按照脚本中的顺序）
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
    
    # 每个run的指标名称（不带mean±std）
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
    
    # 构建表头
    header = ["Dataset", "Method"]
    
    # 添加mean±std列
    header.extend(metric_names)
    
    # 如果指定了num_runs，添加每个run的指标列
    if num_runs is not None and num_runs > 0:
        for run_num in range(1, num_runs + 1):
            for metric in metric_names_raw:
                header.append(f"Run{run_num} {metric}")
    
    return ",".join(header)


def aggregate_metrics_csvs(samples_root, output_csv_path, pattern="scDiffusion_3000"):
    """
    汇总所有MOA的metrics CSV文件到一个汇总文件中。
    
    Args:
        samples_root: samples根目录，例如 samples/fig2/task1_unseen_moa_same
        output_csv_path: 输出汇总CSV文件的路径
        pattern: 要匹配的子目录模式，例如 "scDiffusion_3000"
        
    Returns:
        汇总后的CSV文件绝对路径
    """
    samples_root = Path(samples_root).resolve()
    output_csv_path = Path(output_csv_path).resolve()
    
    # 查找所有匹配的CSV文件
    csv_pattern = str(samples_root / "*" / pattern / "metrics" / "metrics_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"[ERROR] No CSV files found matching pattern: {csv_pattern}")
        return None
    
    print(f"Found {len(csv_files)} CSV files to aggregate")
    
    # 收集所有数据行
    aggregated_rows = []
    moa_names = []
    num_runs = None
    row_has_dataset_col = True  # 数据行是否已含 Dataset 列
    
    for csv_file in sorted(csv_files):
        # 从路径中提取MOA名称
        # 路径格式: .../samples_root/MOA_NAME/scDiffusion_3000/metrics/metrics_MOA_NAME.csv
        parts = Path(csv_file).parts
        try:
            # 找到samples_root在路径中的位置
            root_idx = None
            for i, part in enumerate(parts):
                if samples_root.name in part or str(samples_root) in '/'.join(parts[:i+1]):
                    root_idx = i
                    break
            
            if root_idx is not None and root_idx + 1 < len(parts):
                moa_name = parts[root_idx + 1]
            else:
                # 备用方法：从文件名提取
                filename = Path(csv_file).stem
                moa_name = filename.replace('metrics_', '').replace('_test', '')
            
            latest_row = get_latest_row_from_csv(csv_file)
            if latest_row:
                # 从第一行数据推断列数和运行次数
                if num_runs is None:
                    num_cols = len(latest_row.split(','))
                    # 总列数 = 2 (Dataset + Method) + 11 (mean±std) + num_runs * 11
                    # 若每行只有 Method + 11 + num_runs*11（无 Dataset），则 num_cols = 1+11+11*num_runs
                    if num_cols >= 13 and (num_cols - 1 - 11) % 11 == 0:
                        num_runs = (num_cols - 1 - 11) // 11  # 无 Dataset 列
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
    
    # 确保输出目录存在
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入汇总CSV文件
    try:
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            # 写入表头
            # num_runs为0表示只有mean±std，None表示未检测到
            if num_runs is not None:
                header = generate_csv_header(num_runs if num_runs > 0 else None)
                f.write(header + '\n')
            else:
                print("[WARN] Could not generate header, writing without header")
            
            # 写入所有数据行
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
        description="汇总多个MOA实验结果的CSV文件到一个汇总CSV文件中"
    )
    parser.add_argument(
        "--samples-root",
        type=str,
        required=True,
        help="samples根目录路径，例如 samples/fig2/task1_unseen_moa_same"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="输出汇总CSV文件的路径"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="scDiffusion_3000",
        help="要匹配的子目录模式（默认: scDiffusion_3000）"
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
