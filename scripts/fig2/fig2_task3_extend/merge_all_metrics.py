#!/usr/bin/env python3
"""
 allfig2_task3_extend CSVfileto CSVfile .

each.sh willas4 types (mouse, pig, rabbit, rat) CSVfile.
 willfoundall CSVfileand and.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional

# repo root
SCRIPT_DIR = Path(__file__).parent.absolute()
HOMEDIR = SCRIPT_DIR.parent.parent.parent
SAMPLES_ROOT = HOMEDIR / "samples" / "fig2" / "task3_extend"

# all types
ALL_SPECIES = ["mouse", "pig", "rabbit", "rat"]

# CSVfilepath : name -> filepath 
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
        "has_species_col": True, # filealreadycontainall types, Speciescols
        "is_global": True, # file, each types 
    },
    "Squidiff": {
        "pattern": "{species}/squidiff_1000/metrics_Leave1out_test_{species}.csv",
        "has_species_col": False,
    },
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    normalizedcolsname, scDiffusion colsnameconvert to .
    """
    # colsname 
    column_mapping = {}
    for col in df.columns:
        new_col = col
        # handlescDiffusion colsname
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
     CSVfileandnormalized .
    
    Args:
        file_path: CSVfilepath
        method_name: name 
        species: typesname ( CSVfile containSpeciescols)
    
    Returns:
        normalizedafterDataFrame, file exist returnNone
    """
    if not file_path.exists():
        print(f"[WARNING] CSV file not found: {file_path}", file=sys.stderr)
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # normalizedcolsname
        df = normalize_column_names(df)
        
        # CSVfileno Speciescols, wemustadd
        if species and "Species" not in df.columns:
            df.insert(1, "Species", species)
        
        # EnsureMethodcolsexist correct
        if "Method" in df.columns:
            # Methodcolsvalueand name , 
            df["Method"] = method_name
        else:
            # Methodcols exist, add 
            if "Species" in df.columns:
                df.insert(2, "Method", method_name)
            else:
                df.insert(1, "Method", method_name)
        
        return df
    
    except Exception as e:
        print(f"[ERROR] CSVfile {file_path}: {e}", file=sys.stderr)
        return None


def collect_all_csvs() -> tuple[List[pd.DataFrame], List[tuple[str, str]]]:
    """
     allCSVfileandreturnDataFramecols filecols .
    
    Returns:
        (DataFramecols , filecols [(method_name, species_or_path)])
    """
    all_dfs = []
    missing_files = []
    
    for method_name, config in CSV_PATTERNS.items():
        pattern = config["pattern"]
        is_global = config.get("is_global", False)
        has_species_col = config.get("has_species_col", False)
        
        if is_global:
            # handle file ( scDiffusion)
            csv_path = SAMPLES_ROOT / pattern
            df = read_csv_file(csv_path, method_name)
            if df is not None:
                all_dfs.append(df)
                print(f"[INFO] CSV: {csv_path} ({len(df)} )")
            else:
                missing_files.append((method_name, str(csv_path)))
        else:
            # handleeach typesfile
            for species in ALL_SPECIES:
                csv_path = SAMPLES_ROOT / pattern.format(species=species)
                df = read_csv_file(csv_path, method_name, species=species if not has_species_col else None)
                if df is not None:
                    all_dfs.append(df)
                    print(f"[INFO] CSV: {csv_path} ({len(df)} )")
                else:
                    missing_files.append((method_name, species))
    
    return all_dfs, missing_files


def merge_csvs(output_path: Path):
    """
     andallCSVfileand tooutputfile.
    
    Args:
        output_path: outputCSVfilepath
    """
    print(f"[INFO] Start CSVfile...")
    all_dfs, missing_files = collect_all_csvs()
    
    if not all_dfs:
        print("[ERROR] nofound CSVfile!", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] found {len(all_dfs)} CSVfile")
    
    # file
    if missing_files:
        print(f"\n[WARNING] Found {len(missing_files)} CSVfile:")
        for method_name, species_or_path in missing_files:
            if "/" in species_or_path or "\\" in species_or_path:
                # path
                print(f"  - {method_name}: {species_or_path}")
            else:
                # typesname
                print(f"  - {method_name} ({species_or_path})")
        print("")
    
    # andallDataFrame
    print("[INFO] andCSVfile...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # normalizedcols : Dataset, Species ( ), Method, after all cols
    base_cols = ["Dataset"]
    if "Species" in merged_df.columns:
        base_cols.append("Species")
    base_cols.append("Method")
    
    # getothercols ( cols)
    other_cols = [col for col in merged_df.columns if col not in base_cols]
    
    # define 
    metric_order = [
        "PDS", "MAE", "DES", "E-Distance", "MMD", "R2",
        "Pearson (all genes)", "Pearson Delta (all genes)",
        "Pearson Delta (top 20 DE genes)", "Pearson Delta (top 50 DE genes)",
        "Pearson Delta (top 100 DE genes)"
    ]
    
    # mean±stdcols Runcols
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
    
    # permetric_order mean±stdcols
    def get_metric_priority(col):
        for i, metric in enumerate(metric_order):
            if metric in col:
                return i
        return len(metric_order)
    
    mean_std_cols.sort(key=get_metric_priority)
    
    # perRun id metric Runcols
    def get_run_priority(col):
        # Run id
        import re
        run_match = re.match(r"Run(\d+)", col)
        run_num = int(run_match.group(1)) if run_match else 999
        # metric level
        metric_priority = get_metric_priority(col)
        return (run_num, metric_priority)
    
    run_cols.sort(key=get_run_priority)
    
    # colscols 
    merged_df = merged_df[base_cols + mean_std_cols + run_cols + other_remaining]
    
    # andafterCSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"[INFO] anddone!")
    print(f"[INFO] {len(merged_df)} data")
    print(f"[INFO] outputfile: {output_path}")
    
    # mustbatch info
    print("\n[INFO] must:")
    if "Species" in merged_df.columns:
        summary = merged_df.groupby(["Method", "Species"]).size()
        print(summary.to_string())
        
        # checkeach whether 4 types
        print("\n[INFO] types check:")
        for method in merged_df["Method"].unique():
            species_count = len(merged_df[merged_df["Method"] == method]["Species"].unique())
            expected = 4
            if species_count < expected:
                missing_species = set(ALL_SPECIES) - set(merged_df[merged_df["Method"] == method]["Species"].unique())
                print(f" - {method}: {species_count}/{expected} types ( : {', '.join(missing_species)})")
            else:
                print(f" - {method}: {species_count}/{expected} types ✓")
    else:
        print(merged_df.groupby("Method").size().to_string())


def main():
    """ count"""
    import argparse
    
    # using amountbefore 
    global SAMPLES_ROOT
    
    parser = argparse.ArgumentParser(
        description=" allfig2_task3_extend CSVfile"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(SAMPLES_ROOT / "metrics_all_methods.csv"),
        help="outputCSVfilepath (default: samples/fig2/task3_extend/metrics_all_methods.csv)",
    )
    parser.add_argument(
        "--samples-root",
        type=str,
        default=None,
        help="samples directorypath (default: )",
    )
    
    args = parser.parse_args()
    
    # specifysamples-root, using 
    if args.samples_root:
        SAMPLES_ROOT = Path(args.samples_root)
    
    output_path = Path(args.output)
    
    merge_csvs(output_path)


if __name__ == "__main__":
    main()
