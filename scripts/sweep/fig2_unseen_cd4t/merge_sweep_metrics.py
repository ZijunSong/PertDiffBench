#!/usr/bin/env python3
"""Merge Fig2 unseen-CD4T hyperparameter sweep metrics into one CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path("/data/ppnm/data/PertDiffBench/samples/sweep/fig2_unseen_cd4t")
CONFIG_ROOT = Path(
    "/data/ppnm/PertDiffBench/configs/baselines/sweep/fig2_unseen_cd4t/generated"
)
OUT = ROOT / "fig2_unseen_cd4t_all_sweep_metrics.csv"

EXPECTED = {
    "lr": ["5e-6", "1e-5", "2e-5"],
    "bs": ["1024", "2048", "4096"],
    "steps": ["500", "1000", "2000"],
    "beta": ["1e-4_0.01", "1e-4_0.02", "1e-4_0.04"],
}
METHODS = ["ddpm", "ddpm_mlp"]


def load_hyperparams(run_id: str) -> dict[str, str]:
    cfg_path = CONFIG_ROOT / f"{run_id}.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    if "model" in cfg and "ae" in cfg["model"]:
        diff = cfg["model"]["diffusion"]
    else:
        diff = cfg["diffusion"]

    return {
        "lr": str(cfg["train"]["lr"]),
        "batch_size": str(cfg["train"]["batch_size"]),
        "steps": str(diff["timesteps"]),
        "beta_1": str(diff["beta_1"]),
        "beta_T": str(diff["beta_T"]),
    }


def main() -> None:
    rows: list[dict[str, str]] = []
    missing: list[str] = []

    for axis, tags in EXPECTED.items():
        for method in METHODS:
            for tag in tags:
                run_id = f"{method}__{axis}__{tag}"
                csv_path = ROOT / run_id / "metrics_CD4T_p0.25.csv"
                if not csv_path.exists():
                    missing.append(str(csv_path))
                    continue

                with csv_path.open(newline="") as f:
                    data = next(csv.DictReader(f))

                hp = load_hyperparams(run_id)
                method_label = "DDPM+MLP" if method == "ddpm_mlp" else "DDPM"
                pds = float(data.get("PDS", "0") or 0)
                eval_ok = "yes" if pds > 0 else "no"

                rows.append(
                    {
                        "run_id": run_id,
                        "method": method_label,
                        "sweep_axis": axis,
                        "sweep_value": tag,
                        "lr": hp["lr"],
                        "batch_size": hp["batch_size"],
                        "steps": hp["steps"],
                        "beta_1": hp["beta_1"],
                        "beta_T": hp["beta_T"],
                        "holdout": "CD4T",
                        "control_ratio": "p0.25",
                        "eval_ok": eval_ok,
                        "Method": data.get("Method", ""),
                        "PDS": data["PDS"],
                        "MAE": data["MAE"],
                        "DES": data["DES"],
                        "E-Distance": data["E-Distance"],
                        "MMD": data["MMD"],
                        "R2": data["R2"],
                        "Pearson_all": data["Pearson_all"],
                        "PearsonDelta_all": data["PearsonDelta_all"],
                        "PearsonDelta_DE20": data["PearsonDelta_DE20"],
                        "PearsonDelta_DE50": data["PearsonDelta_DE50"],
                        "PearsonDelta_DE100": data["PearsonDelta_DE100"],
                    }
                )

    order_axis = {"lr": 0, "bs": 1, "steps": 2, "beta": 3}
    order_method = {"DDPM": 0, "DDPM+MLP": 1}
    rows.sort(
        key=lambda x: (
            order_axis[x["sweep_axis"]],
            order_method[x["method"]],
            x["sweep_value"],
        )
    )

    fieldnames = [
        "run_id",
        "method",
        "sweep_axis",
        "sweep_value",
        "lr",
        "batch_size",
        "steps",
        "beta_1",
        "beta_T",
        "holdout",
        "control_ratio",
        "eval_ok",
        "Method",
        "PDS",
        "MAE",
        "DES",
        "E-Distance",
        "MMD",
        "R2",
        "Pearson_all",
        "PearsonDelta_all",
        "PearsonDelta_DE20",
        "PearsonDelta_DE50",
        "PearsonDelta_DE100",
    ]

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(rows)} runs -> {OUT}")
    if missing:
        print(f"Missing {len(missing)} CSV files:")
        for path in missing:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
