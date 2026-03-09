#!/usr/bin/env python3
"""
汇总 noise_exp 下 4 组实验 × 6 个 baseline 的 metrics CSV 为一个大 CSV。
4 组实验: gaussian_perturbed_data, lognormal_bionoise_perturbed_data,
         poisson_technoise_perturbed_data, zero_inflation_technoise_perturbed_data
6 个 baseline: ddpm, ddpm_mlp, scdiff, scdiffusion, scgen, squidiff
若某 (实验, baseline) 无结果文件，则对应行留空（仅保留 Experiment、Baseline 列）。
输出: samples/noise_exp/noise_exp_metrics_merged.csv
"""
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# 4 组实验 × 6 个 baseline：每个 (实验, baseline) 对应在 samples 下的 CSV 路径规则
# 规则为 (samples 下相对路径的 glob 或包含子串)，用于在该目录下递归找 metrics_*.csv
EXPERIMENTS = [
    "gaussian_perturbed_data",
    "lognormal_bionoise_perturbed_data",
    "poisson_technoise_perturbed_data",
    "zero_inflation_technoise_perturbed_data",
]
# Baseline 顺序：Squidiff, scDiff, scDiffusion, scGen, DDPM, DDPM+MLP（内部名 → 显示名）
BASELINE_ORDER = ["squidiff", "scdiff", "scdiffusion", "scgen", "ddpm", "ddpm_mlp"]
BASELINE_DISPLAY = {
    "squidiff": "Squidiff",
    "scdiff": "scDiff",
    "scdiffusion": "scDiffusion",
    "scgen": "scGen",
    "ddpm": "DDPM",
    "ddpm_mlp": "DDPM+MLP",
}
BASELINES = BASELINE_ORDER

# 指标列顺序：PDS, MAE, DES, E-Distance, MMD, R2, Pearson (all genes)↑, Pearson Delta (all genes)↑,
# Pearson Delta (top 20 DE genes)↑, Pearson Delta (top 50 DE genes)↑, Pearson Delta (top 100 DE genes)↑
METRIC_MEAN_STD = [
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
    "Pearson Delta (top 100 DE genes) (mean±std)",
]
METRIC_RUN_PREFIX = ["Run1", "Run2", "Run3"]
METRIC_NAMES_RUN = [
    "PDS", "MAE", "DES", "E-Distance", "MMD", "R2",
    "Pearson (all genes)", "Pearson Delta (all genes)",
    "Pearson Delta (top 20 DE genes)", "Pearson Delta (top 50 DE genes)", "Pearson Delta (top 100 DE genes)",
]


def _metric_columns_order():
    """返回期望的指标列顺序（含 mean±std 与 Run1/Run2/Run3 的 11 项）。"""
    out = list(METRIC_MEAN_STD)
    for prefix in METRIC_RUN_PREFIX:
        for name in METRIC_NAMES_RUN:
            out.append(f"{prefix} {name}")
    return out

# 每个 (实验, baseline) 在 samples 下的根目录及用于识别该 baseline 的路径子串
# 结构: (experiment_name, baseline_name) -> (samples_subdir, path_contains)
# 在 samples_subdir 下递归找 *metrics*.csv，且路径包含 path_contains 的归为该 baseline
NOISE_SAMPLES_ROOTS = {
    "gaussian_perturbed_data": [
        ("samples/gaussian_noise", None),   # 该目录下仅 ddpm/ddpm_mlp，用目录名区分
        ("samples/fig1/task1", None),      # scgen, scdiff, scdiffusion（squidiff 不写 CSV）
    ],
    "lognormal_bionoise_perturbed_data": [("samples/lognormal_bionoise", None)],
    "poisson_technoise_perturbed_data": [("samples/poisson_technoise", None)],
    "zero_inflation_technoise_perturbed_data": [("samples/zero_inflation_technoise", None)],
}

# 从相对路径推断 baseline：匹配顺序即优先级
BASELINE_PATH_MARKERS = [
    ("scrna_ddpm_scrna", "ddpm"),
    ("mlp_ddpm_mlp", "ddpm_mlp"),
    ("scDiffusion", "scdiffusion"),
    ("scdiff", "scdiff"),
    ("scgen", "scgen"),
    ("squidiff", "squidiff"),
]


def infer_baseline(rel_path: str, method_value: str | None = None) -> str | None:
    # gaussian_noise 下 scrna_ddpm_scrna 同时被 ddpm 与 ddpm_mlp 使用，用 Method 区分
    if "scrna_ddpm_scrna" in rel_path and "gaussian_noise" in rel_path and method_value:
        if "MLP" in method_value or "6998" in method_value or "scRNA-DDPM-scRNA-" in method_value:
            return "ddpm_mlp"
        return "ddpm"
    for marker, baseline in BASELINE_PATH_MARKERS:
        if marker in rel_path:
            return baseline
    return None


def infer_experiment(rel_path: str) -> str | None:
    if "gaussian_noise" in rel_path:
        return "gaussian_perturbed_data"
    if "fig1/task1" in rel_path and ("scgen" in rel_path or "scdiff" in rel_path or "scDiffusion" in rel_path):
        return "gaussian_perturbed_data"
    if "lognormal_bionoise" in rel_path:
        return "lognormal_bionoise_perturbed_data"
    if "poisson_technoise" in rel_path:
        return "poisson_technoise_perturbed_data"
    if "zero_inflation_technoise" in rel_path:
        return "zero_inflation_technoise_perturbed_data"
    return None


def _read_method_from_csv(csv_path: Path) -> str | None:
    try:
        df = pd.read_csv(csv_path, nrows=1)
        if "Method" in df.columns and len(df) > 0 and pd.notna(df["Method"].iloc[0]):
            return str(df["Method"].iloc[0])
    except Exception:
        pass
    return None


def collect_csv_paths() -> list[tuple[str, str, Path]]:
    """返回 [(experiment, baseline, csv_path), ...]"""
    out = []
    for exp_name, roots in NOISE_SAMPLES_ROOTS.items():
        for root_spec in roots:
            root_dir = REPO_ROOT / root_spec[0]
            if not root_dir.is_dir():
                continue
            for csv_path in root_dir.rglob("metrics_*.csv"):
                try:
                    rel = csv_path.relative_to(REPO_ROOT)
                    rel_str = str(rel.as_posix())
                    inferred_exp = infer_experiment(rel_str)
                    if inferred_exp not in EXPERIMENTS:
                        continue
                    method_val = None
                    if "gaussian_noise" in rel_str and "scrna_ddpm_scrna" in rel_str:
                        method_val = _read_method_from_csv(csv_path)
                    baseline = infer_baseline(rel_str, method_val)
                    if baseline:
                        out.append((inferred_exp, baseline, csv_path))
                except ValueError:
                    continue
    return out


def main():
    samples_noise = REPO_ROOT / "samples" / "noise_exp"
    samples_noise.mkdir(parents=True, exist_ok=True)
    out_path = samples_noise / "noise_exp_metrics_merged.csv"

    collected = collect_csv_paths()
    # 按 (exp, baseline) 分组，每个组内可能多个 CSV（不同 cell_type/noise_level）
    from collections import defaultdict
    by_slot = defaultdict(list)
    for exp, baseline, path in collected:
        by_slot[(exp, baseline)].append(path)

    all_dfs = []
    seen_columns = set()

    for exp in EXPERIMENTS:
        for baseline in BASELINES:
            paths = by_slot.get((exp, baseline), [])
            if not paths:
                # 缺结果：加一行空行，仅 Experiment、Baseline 有值
                all_dfs.append(
                    pd.DataFrame([{"Experiment": exp, "Baseline": baseline}])
                )
                continue
            for p in sorted(paths):
                try:
                    df = pd.read_csv(p)
                    if df.empty:
                        continue
                    # 避免 Method/Dataset/Noise 与后续合并冲突，先加两列
                    if "Experiment" not in df.columns:
                        df.insert(0, "Experiment", exp)
                    else:
                        df["Experiment"] = exp
                    if "Baseline" not in df.columns:
                        df.insert(1, "Baseline", baseline)
                    else:
                        df["Baseline"] = baseline
                    all_dfs.append(df)
                    seen_columns.update(df.columns.tolist())
                except Exception as e:
                    print(f"[WARN] 跳过 {p}: {e}")

    if not all_dfs:
        pd.DataFrame(columns=["Experiment", "Baseline"]).to_csv(out_path, index=False)
        print("未找到任何 metrics CSV，已写出空表。")
        print(f"路径: {out_path.resolve()}")
        return

    merged = pd.concat(all_dfs, axis=0, ignore_index=True, sort=False)
    # 行顺序：先按 Experiment（固定顺序），再按 Baseline（Squidiff → scDiff → … → DDPM+MLP）
    exp_order = {e: i for i, e in enumerate(EXPERIMENTS)}
    baseline_order = {b: i for i, b in enumerate(BASELINE_ORDER)}
    merged = merged.sort_values(
        by=["Experiment", "Baseline"],
        key=lambda s: s.map(exp_order if s.name == "Experiment" else baseline_order),
    ).reset_index(drop=True)
    # Baseline 列改为显示名
    merged["Baseline"] = merged["Baseline"].map(lambda b: BASELINE_DISPLAY.get(b, b))
    # 列顺序：Experiment, Baseline, Dataset, Noise, Method，再按指标顺序（PDS, MAE, DES, …），最后其余列
    prefix_cols = ["Experiment", "Baseline", "Dataset", "Noise", "Method"]
    metric_order = _metric_columns_order()
    ordered = [c for c in prefix_cols if c in merged.columns]
    for c in metric_order:
        if c in merged.columns and c not in ordered:
            ordered.append(c)
    for c in merged.columns:
        if c not in ordered:
            ordered.append(c)
    merged = merged.reindex(columns=ordered)
    merged.to_csv(out_path, index=False)
    total_rows = len(merged)
    metric_col = "PDS (mean±std)" if "PDS (mean±std)" in merged.columns else "Method"
    filled = sum(1 for _, r in merged.iterrows() if pd.notna(r.get(metric_col, None)))
    print(f"已汇总 4×6 组实验：共 {total_rows} 行（含缺失留空），其中有效结果约 {filled} 行 -> {out_path}")
    print(f"绝对路径: {out_path.resolve()}")


if __name__ == "__main__":
    main()
