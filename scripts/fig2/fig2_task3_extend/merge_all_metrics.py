#!/usr/bin/env python3
"""
汇总所有fig2_task3_extend脚本生成的CSV文件到一个统一的CSV文件中。

每个.sh脚本会为4个物种（mouse, pig, rabbit, rat）各生成一个CSV文件。
此脚本会找到所有这些CSV文件并将它们合并。
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional

# 项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
HOMEDIR = SCRIPT_DIR.parent.parent.parent
SAMPLES_ROOT = HOMEDIR / "samples" / "fig2" / "task3_extend"

# 所有物种
ALL_SPECIES = ["mouse", "pig", "rabbit", "rat"]

# CSV文件路径映射：方法名 -> 文件路径模式
CSV_PATTERNS = {
    "scRNA-DDPM-scRNA": {
        "pattern": "{species}/scrna_ddpm_scrna/metrics_leave1out_{species}.csv",
        "has_species_col": False,
    },
    "MLP-DDPM-MLP": {
        "pattern": "{species}/mlp_ddpm_mlp/metrics_leave1out_{species}.csv",
        "has_species_col": False,
    },
    "scGen": {
        "pattern": "{species}/scgen/metrics_Leave1out_test_{species}.csv",
        "has_species_col": False,
    },
    "scDiff": {
        "pattern": "scdiff/{species}/metrics_leave1out_{species}.csv",
        "has_species_col": False,
    },
    "scDiffusion(6619)": {
        "pattern": "scDiffusion_6619/metrics_all.csv",
        "has_species_col": True,  # 这个文件已经包含了所有物种，且有Species列
        "is_global": True,  # 这是一个全局文件，不是每个物种一个
    },
    "Squidiff": {
        "pattern": "{species}/squidiff_1000/metrics_Leave1out_test_{species}.csv",
        "has_species_col": False,
    },
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名，将scDiffusion的特殊列名转换为标准格式。
    """
    # 创建列名映射
    column_mapping = {}
    for col in df.columns:
        new_col = col
        # 处理scDiffusion的特殊列名
        if "Pearson (all)" in col and "Pearson (all genes)" not in col:
            new_col = col.replace("Pearson (all)", "Pearson (all genes)")
        if "PearsonΔ" in col:
            new_col = col.replace("PearsonΔ", "Pearson Delta")
        if "PearsonΔ(all)" in col:
            new_col = col.replace("PearsonΔ(all)", "Pearson Delta (all genes)")
        if "PearsonΔ(top20)" in col:
            new_col = col.replace("PearsonΔ(top20)", "Pearson Delta (top 20 DE genes)")
        if "PearsonΔ(top50)" in col:
            new_col = col.replace("PearsonΔ(top50)", "Pearson Delta (top 50 DE genes)")
        if "PearsonΔ(top100)" in col:
            new_col = col.replace("PearsonΔ(top100)", "Pearson Delta (top 100 DE genes)")
        if "Run" in col and "Pearson(all)" in col:
            new_col = col.replace("Pearson(all)", "Pearson (all genes)")
        column_mapping[col] = new_col
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df


def read_csv_file(file_path: Path, method_name: str, species: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    读取CSV文件并标准化格式。
    
    Args:
        file_path: CSV文件路径
        method_name: 方法名称
        species: 物种名称（如果CSV文件不包含Species列）
    
    Returns:
        标准化后的DataFrame，如果文件不存在则返回None
    """
    if not file_path.exists():
        print(f"[WARNING] CSV文件不存在: {file_path}", file=sys.stderr)
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # 标准化列名
        df = normalize_column_names(df)
        
        # 如果CSV文件没有Species列，但我们需要添加
        if species and "Species" not in df.columns:
            df.insert(1, "Species", species)
        
        # 确保Method列存在且正确
        if "Method" in df.columns:
            # 如果Method列的值与方法名不一致，更新它
            df["Method"] = method_name
        else:
            # 如果Method列不存在，添加它
            if "Species" in df.columns:
                df.insert(2, "Method", method_name)
            else:
                df.insert(1, "Method", method_name)
        
        return df
    
    except Exception as e:
        print(f"[ERROR] 读取CSV文件失败 {file_path}: {e}", file=sys.stderr)
        return None


def collect_all_csvs() -> tuple[List[pd.DataFrame], List[tuple[str, str]]]:
    """
    收集所有CSV文件并返回DataFrame列表和缺失文件列表。
    
    Returns:
        (DataFrame列表, 缺失文件列表[(method_name, species_or_path)])
    """
    all_dfs = []
    missing_files = []
    
    for method_name, config in CSV_PATTERNS.items():
        pattern = config["pattern"]
        is_global = config.get("is_global", False)
        has_species_col = config.get("has_species_col", False)
        
        if is_global:
            # 处理全局文件（如scDiffusion）
            csv_path = SAMPLES_ROOT / pattern
            df = read_csv_file(csv_path, method_name)
            if df is not None:
                all_dfs.append(df)
                print(f"[INFO] 读取全局CSV: {csv_path} ({len(df)} 行)")
            else:
                missing_files.append((method_name, str(csv_path)))
        else:
            # 处理每个物种的文件
            for species in ALL_SPECIES:
                csv_path = SAMPLES_ROOT / pattern.format(species=species)
                df = read_csv_file(csv_path, method_name, species=species if not has_species_col else None)
                if df is not None:
                    all_dfs.append(df)
                    print(f"[INFO] 读取CSV: {csv_path} ({len(df)} 行)")
                else:
                    missing_files.append((method_name, species))
    
    return all_dfs, missing_files


def merge_csvs(output_path: Path):
    """
    合并所有CSV文件并保存到输出文件。
    
    Args:
        output_path: 输出CSV文件路径
    """
    print(f"[INFO] 开始收集CSV文件...")
    all_dfs, missing_files = collect_all_csvs()
    
    if not all_dfs:
        print("[ERROR] 没有找到任何CSV文件！", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] 找到 {len(all_dfs)} 个CSV文件")
    
    # 报告缺失的文件
    if missing_files:
        print(f"\n[WARNING] 发现 {len(missing_files)} 个缺失的CSV文件:")
        for method_name, species_or_path in missing_files:
            if "/" in species_or_path or "\\" in species_or_path:
                # 这是一个路径
                print(f"  - {method_name}: {species_or_path}")
            else:
                # 这是一个物种名
                print(f"  - {method_name} ({species_or_path})")
        print("")
    
    # 合并所有DataFrame
    print("[INFO] 合并CSV文件...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # 标准化列顺序：Dataset, Species (如果有), Method, 然后是所有指标列
    base_cols = ["Dataset"]
    if "Species" in merged_df.columns:
        base_cols.append("Species")
    base_cols.append("Method")
    
    # 获取其他列（指标列）
    other_cols = [col for col in merged_df.columns if col not in base_cols]
    
    # 定义指标的标准顺序
    metric_order = [
        "PDS", "MAE", "DES", "E-Distance", "MMD", "R2",
        "Pearson (all genes)", "Pearson Delta (all genes)",
        "Pearson Delta (top 20 DE genes)", "Pearson Delta (top 50 DE genes)",
        "Pearson Delta (top 100 DE genes)"
    ]
    
    # 分离mean±std列和Run列
    mean_std_cols = []
    run_cols = []
    other_remaining = []
    
    for col in other_cols:
        if " (mean±std)" in col:
            mean_std_cols.append(col)
        elif col.startswith("Run"):
            run_cols.append(col)
        else:
            other_remaining.append(col)
    
    # 按照metric_order排序mean±std列
    def get_metric_priority(col):
        for i, metric in enumerate(metric_order):
            if metric in col:
                return i
        return len(metric_order)
    
    mean_std_cols.sort(key=get_metric_priority)
    
    # 按照Run编号和metric顺序排序Run列
    def get_run_priority(col):
        # 提取Run编号
        import re
        run_match = re.match(r"Run(\d+)", col)
        run_num = int(run_match.group(1)) if run_match else 999
        # 提取metric优先级
        metric_priority = get_metric_priority(col)
        return (run_num, metric_priority)
    
    run_cols.sort(key=get_run_priority)
    
    # 重新排列列顺序
    merged_df = merged_df[base_cols + mean_std_cols + run_cols + other_remaining]
    
    # 保存合并后的CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"[INFO] 合并完成！")
    print(f"[INFO] 总共 {len(merged_df)} 行数据")
    print(f"[INFO] 输出文件: {output_path}")
    
    # 打印摘要信息
    print("\n[INFO] 汇总摘要:")
    if "Species" in merged_df.columns:
        summary = merged_df.groupby(["Method", "Species"]).size()
        print(summary.to_string())
        
        # 检查每个方法是否都有4个物种
        print("\n[INFO] 各方法物种完整性检查:")
        for method in merged_df["Method"].unique():
            species_count = len(merged_df[merged_df["Method"] == method]["Species"].unique())
            expected = 4
            if species_count < expected:
                missing_species = set(ALL_SPECIES) - set(merged_df[merged_df["Method"] == method]["Species"].unique())
                print(f"  - {method}: {species_count}/{expected} 物种 (缺失: {', '.join(missing_species)})")
            else:
                print(f"  - {method}: {species_count}/{expected} 物种 ✓")
    else:
        print(merged_df.groupby("Method").size().to_string())


def main():
    """主函数"""
    import argparse
    
    # 使用全局变量前先声明
    global SAMPLES_ROOT
    
    parser = argparse.ArgumentParser(
        description="汇总所有fig2_task3_extend脚本生成的CSV文件"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(SAMPLES_ROOT / "metrics_all_methods.csv"),
        help="输出CSV文件路径（默认: samples/fig2/task3_extend/metrics_all_methods.csv）",
    )
    parser.add_argument(
        "--samples-root",
        type=str,
        default=None,
        help="samples根目录路径（默认: 自动检测）",
    )
    
    args = parser.parse_args()
    
    # 如果指定了samples-root，使用它
    if args.samples_root:
        SAMPLES_ROOT = Path(args.samples_root)
    
    output_path = Path(args.output)
    
    merge_csvs(output_path)


if __name__ == "__main__":
    main()
