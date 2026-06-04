#!/usr/bin/env python3
"""
from scDiff train logdir (with checkpoints/last.ckpt) test, 
parse stdout 11 , per-dataset CSV, andoptional CSV.

using (inrepo root):
  # test, samples/fig2/task1_unseen_moa_diff/*/scdiff/metrics/*.csv
  python scripts/fig2/fig2_task1_moa/rerun_test_only_scdiff_moa_diff.py \
    --logdir-root logs \
    --datadir-substr unseen_diff_moa \
    --samples-root samples/fig2/task1_unseen_moa_diff \
    --method-name "scDiff(v7.5)" \
    --num-runs 3

  # when CSV
  python scripts/fig2/fig2_task1_moa/rerun_test_only_scdiff_moa_diff.py \
    ... --aggregate-to samples/fig2/task1_unseen_moa_diff/aggregated_metrics_scdiff.csv
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# and shell AWK 11 name (for parse CSV)
METRIC_PATTERNS = [
    (r"Perturbation Discrimination Score \(PDS\):\s*([\d.eE+-]+)", "PDS"),
    (r"Mean Absolute Error \(MAE\):\s*([\d.eE+-]+)", "MAE"),
    (r"Differential Expression Score \(DES\):\s*([\d.eE+-]+)", "DES"),
    (r"E-Distance:\s*([\d.eE+-]+)", "E-Distance"),
    (r"Maximum Mean Discrepancy \(MMD\):\s*([\d.eE+-]+)", "MMD"),
    (r"R-squared \(R2\):\s*([\d.eE+-]+)", "R2"),
    (r"Pearson \(all genes\):\s*([\d.eE+-]+)", "Pearson (all genes)"),
    (r"Pearson Delta \(all genes\):\s*([\d.eE+-]+)", "Pearson Delta (all genes)"),
    (r"Pearson Delta \(top 20 DE genes\):\s*([\d.eE+-]+)", "Pearson Delta (top 20 DE genes)"),
    (r"Pearson Delta \(top 50 DE genes\):\s*([\d.eE+-]+)", "Pearson Delta (top 50 DE genes)"),
    (r"Pearson Delta \(top 100 DE genes\):\s*([\d.eE+-]+)", "Pearson Delta (top 100 DE genes)"),
]


def parse_metrics_from_stdout(text: str) -> dict:
    """from test stdout parse 11 , return {metric_short_name: float}."""
    values = {}
    for pattern, short_name in METRIC_PATTERNS:
        m = re.search(pattern, text)
        if m:
            try:
                values[short_name] = float(m.group(1))
            except ValueError:
                values[short_name] = float("nan")
        else:
            values[short_name] = float("nan")
    return values


def get_dataset_from_logdir(logdir: Path) -> str | None:
    """from logdir/configs/*.yaml data.params.test.params.dataset (or train)."""
    cfg_dir = logdir / "configs"
    if not cfg_dir.is_dir():
        return None
    # project yaml data 
    for f in sorted(cfg_dir.glob("*.yaml")):
        try:
            with open(f, "r") as fp:
                data = yaml.safe_load(fp)
            if not data:
                continue
            # data.params.test.params.dataset or data.params.train.params.dataset
            for key in ("test", "train"):
                params = (data.get("data") or {}).get("params") or {}
                t = (params.get(key) or {}).get("params") or {}
                if isinstance(t, dict) and "dataset" in t:
                    return t["dataset"]
        except Exception:
            continue
    return None


def get_datadir_from_logdir(logdir: Path) -> str | None:
    """from logdir config datadir (test or train)."""
    cfg_dir = logdir / "configs"
    if not cfg_dir.is_dir():
        return None
    for f in sorted(cfg_dir.glob("*.yaml")):
        try:
            with open(f, "r") as fp:
                data = yaml.safe_load(fp)
            if not data:
                continue
            for key in ("test", "train"):
                params = (data.get("data") or {}).get("params") or {}
                t = (params.get(key) or {}).get("params") or {}
                if isinstance(t, dict) and "datadir" in t:
                    return t["datadir"]
        except Exception:
            continue
    return None


def find_logdirs(logdir_root: Path, datadir_substr: str, num_runs: int) -> dict[str, list[Path]]:
    """
     logdir_root, allwith checkpoints/last.ckpt config datadir contain datadir_substr directory.
     dataset , mtime before num_runs ( as run1..runN).
    return {dataset: [logdir_run1, logdir_run2, ...]}.
    """
    logdir_root = Path(logdir_root).resolve()
    by_dataset: dict[str, list[tuple[float, Path]]] = {}

    for d in logdir_root.iterdir():
        if not d.is_dir():
            continue
        ckpt = d / "checkpoints" / "last.ckpt"
        if not ckpt.is_file():
            continue
        datadir = get_datadir_from_logdir(d)
        if not datadir or datadir_substr not in datadir:
            continue
        dataset = get_dataset_from_logdir(d)
        if not dataset:
            continue
        mtime = d.stat().st_mtime
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append((mtime, d))

    # mtime , before num_runs 
    result = {}
    for dataset, list_mtime_dir in by_dataset.items():
        list_mtime_dir.sort(key=lambda x: x[0])
        chosen = [p for _, p in list_mtime_dir[:num_runs]]
        if len(chosen) == num_runs:
            result[dataset] = chosen
        else:
            print(f"[WARN] Dataset {dataset}: found {len(list_mtime_dir)} runs, need {num_runs}, skipping or using all.")
            result[dataset] = chosen
    return result


def run_test_only(project_root: Path, logdir: Path) -> str:
    """ main.py --resume <logdir> --train False, return stdout+stderr."""
    cmd = [
        sys.executable,
        str(project_root / "src" / "scDiff" / "main.py"),
        "--resume", str(logdir.resolve()),
        "--train", "False",
    ]
    env = os.environ.copy()
    # and fig2 : items + src/scDiff, from scdiff can 
    scdiff_src = str(project_root / "src" / "scDiff")
    env["PYTHONPATH"] = os.pathsep.join([str(project_root), scdiff_src, env.get("PYTHONPATH", "")])
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    return result.stdout + "\n" + result.stderr


def _is_nan(v):
    import math
    return isinstance(v, float) and math.isnan(v)


def write_per_dataset_csv(
    samples_root: Path,
    dataset: str,
    test_ds: str,
    runs_metrics: list[dict],
    method_name: str,
) -> None:
    """ dataset metrics CSV: method + 11 mean±std + Run1..RunN 11 cols."""
    metric_short_names = [m[1] for m in METRIC_PATTERNS]
    n_runs = len(runs_metrics)

    def mean_std(vals):
        valid = [float(x) for x in vals if x is not None and not _is_nan(x)]
        if not valid:
            return 0.0, 0.0
        m = sum(valid) / len(valid)
        if len(valid) > 1:
            s = (sum((x - m) ** 2 for x in valid) / (len(valid) - 1)) ** 0.5
        else:
            s = 0.0
        return m, s

    # header
    header = ["Method"]
    for name in metric_short_names:
        header.append(f"{name} (mean±std)")
    for r in range(1, n_runs + 1):
        for name in metric_short_names:
            header.append(f"Run{r} {name}")

    # countvalue 
    row = [method_name]
    for name in metric_short_names:
        vals = [run.get(name) for run in runs_metrics]
        m, s = mean_std(vals)
        row.append(f"{m:.4f}±{s:.4f}")
    for run in runs_metrics:
        for name in metric_short_names:
            v = run.get(name, float("nan"))
            row.append(f"{v:.4f}" if not _is_nan(v) else "nan")

    out_dir = samples_root / dataset / "scdiff" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"metrics_{test_ds}.csv"
    with open(out_file, "w") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")
    print(f"  Wrote {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Re-run scDiff test only from existing logdirs and write CSVs")
    parser.add_argument("--logdir-root", type=str, default="logs", help="Root directory containing run logdirs")
    parser.add_argument("--datadir-substr", type=str, default="unseen_diff_moa",
                       help="Only consider logdirs whose config datadir contains this string")
    parser.add_argument("--samples-root", type=str, default="samples/fig2/task1_unseen_moa_diff",
                       help="Where to write per-dataset metrics CSVs")
    parser.add_argument("--method-name", type=str, default="scDiff(v7.5)")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--aggregate-to", type=str, default=None,
                       help="If set, write aggregated CSV to this path (after writing per-dataset CSVs)")
    parser.add_argument("--dry-run", action="store_true", help="Only list logdirs, do not run test or write")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    os.chdir(project_root)

    logdir_root = Path(args.logdir_root).resolve()
    samples_root = Path(args.samples_root).resolve()

    dataset_to_logdirs = find_logdirs(logdir_root, args.datadir_substr, args.num_runs)
    if not dataset_to_logdirs:
        print("[ERROR] No logdirs found. Check --logdir-root and --datadir-substr.")
        return 1
    print(f"Found {len(dataset_to_logdirs)} datasets with {args.num_runs} runs each (by mtime).")

    if args.dry_run:
        for ds, dirs in sorted(dataset_to_logdirs.items()):
            print(f"  {ds}: {[str(d) for d in dirs]}")
        return 0

    for dataset in sorted(dataset_to_logdirs.keys()):
        logdirs = dataset_to_logdirs[dataset]
        test_ds = f"{dataset}_test"
        runs_metrics = []
        for idx, logdir in enumerate(logdirs):
            print(f"  Running test for {dataset} run{idx + 1} (resume {logdir}) ...")
            out = run_test_only(project_root, logdir)
            metrics = parse_metrics_from_stdout(out)
            # rows parseto, output 
            if all(_is_nan(metrics.get(m[1], float("nan"))) for m in METRIC_PATTERNS):
                print(f"    [WARN] No metrics parsed for {dataset} run{idx+1}. Last 40 lines of output:")
                for line in out.strip().split("\n")[-40:]:
                    print("    ", line[:200])
            runs_metrics.append(metrics)
        write_per_dataset_csv(samples_root, dataset, test_ds, runs_metrics, args.method_name)

    if args.aggregate_to:
        # : each dataset CSV after , before Dataset cols
        agg_path = Path(args.aggregate_to).resolve()
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        metric_short_names = [m[1] for m in METRIC_PATTERNS]
        header = ["Dataset", "Method"]
        for name in metric_short_names:
            header.append(f"{name} (mean±std)")
        for r in range(1, args.num_runs + 1):
            for name in metric_short_names:
                header.append(f"Run{r} {name}")
        with open(agg_path, "w") as f:
            f.write(",".join(header) + "\n")
            for dataset in sorted(dataset_to_logdirs.keys()):
                csv_path = samples_root / dataset / "scdiff" / "metrics" / f"metrics_{dataset}_test.csv"
                if not csv_path.exists():
                    continue
                lines = [ln.strip() for ln in open(csv_path) if ln.strip()]
                if len(lines) < 2:
                    continue
                data_row = lines[-1]
                f.write(dataset + "," + data_row + "\n")
        print(f"Aggregated CSV: {agg_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
