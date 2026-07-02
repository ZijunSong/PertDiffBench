from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set

from pertdiffbench.tasks.base import TaskSpec

# task -> models that do NOT support it
MODEL_TASK_BLOCKLIST: Dict[str, Set[str]] = {
    "scgen": {"cross_celltype", "temporal"},
    "scdiff": {"cross_celltype", "temporal"},
    "chemcpa": {
        "known_condition",
        "cross_celltype",
        "cross_celltype_extend",
        "cross_celltype_plus",
        "cross_species",
        "cross_species_loo",
        "temporal",
        "noise",
        "encoder",
    },
    "encoder": {
        "known_condition",
        "cross_celltype",
        "cross_celltype_extend",
        "cross_celltype_plus",
        "cross_species",
        "cross_species_loo",
        "moa_same",
        "moa_diff",
        "temporal",
        "noise",
    },
}

MOA_TASKS = {"moa_same", "moa_diff"}
TEMPORAL_TASKS = {"temporal"}
ENCODER_TASKS = {"encoder"}


class PerturbationBackend(ABC):
    name: str
    display_name: str

    def supports(self, spec: TaskSpec) -> bool:
        blocked = MODEL_TASK_BLOCKLIST.get(self.name, set())
        if spec.task_name in blocked:
            return False
        if self.name != "encoder" and spec.task_name in ENCODER_TASKS:
            return False
        if self.name == "encoder" and spec.task_name not in ENCODER_TASKS:
            return False
        if self.name == "chemcpa" and spec.task_name not in MOA_TASKS:
            return False
        return True

    def training_data(self, spec: TaskSpec) -> Path:
        if spec.combined_h5ad and spec.task_name in {
            "cross_celltype_extend",
            "cross_celltype_plus",
        }:
            return spec.combined_h5ad
        return spec.train_h5ad

    def run_dir(self, spec: TaskSpec, run_index: int) -> Path:
        return spec.ckpt_dir / self.name / f"run{run_index}"

    def sample_run_dir(self, spec: TaskSpec, run_index: int) -> Path:
        return spec.sample_dir / self.name / f"run{run_index}"

    @abstractmethod
    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        ...

    @abstractmethod
    def train(self, spec: TaskSpec, run_dir: Path, repo_root: Path, env: dict, run_index: int) -> None:
        ...

    @abstractmethod
    def evaluate(
        self,
        spec: TaskSpec,
        run_dir: Path,
        ckpt_path: Path,
        repo_root: Path,
        env: dict,
        run_index: int,
    ) -> Dict[str, float]:
        ...

    def train_once(self, spec: TaskSpec) -> bool:
        return spec.train_once or spec.task_name in {"cross_species", "cross_species_loo", "noise"}

    def squidiff_train_once(self, spec: TaskSpec) -> bool:
        return spec.squidiff_train_once and spec.task_name == "known_condition"


def run_env_for_index(env: dict, run_index: int) -> dict:
    """Copy env and set RUN_SEED to 0-based run index (0, 1, 2 for NUM_RUNS=3)."""
    out = env.copy()
    out["RUN_SEED"] = str(max(0, run_index - 1))
    return out


def run_python(
    repo_root: Path,
    script: str,
    args: List[str],
    env: dict,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(repo_root / script), *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd or repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_success(proc: subprocess.CompletedProcess, step: str) -> str:
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"{step} failed (exit {proc.returncode}).\n{output[-6000:]}")
    return output


def pair_only_args(spec: TaskSpec) -> List[str]:
    if spec.pair_only_obs_key and spec.pair_only_obs_value:
        return [
            "--pair-only-obs-key", spec.pair_only_obs_key,
            "--pair-only-obs-value", spec.pair_only_obs_value,
        ]
    return []
