from __future__ import annotations

from pathlib import Path
from typing import Dict

from pertdiffbench.backends.base import MOA_TASKS, PerturbationBackend, ensure_success, run_python
from pertdiffbench.evaluate import parse_metrics_from_output
from pertdiffbench.tasks.base import TaskSpec


class ChemCPABackend(PerturbationBackend):
    name = "chemcpa"
    display_name = "ChemCPA"
    config = "configs/chemcpa/moa_fig2_task1.yaml"
    ckpt_name = "last.ckpt"

    def supports(self, spec: TaskSpec) -> bool:
        return spec.task_name in MOA_TASKS

    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        return run_dir / self.ckpt_name

    def _combined_h5ad(self, spec: TaskSpec, run_dir: Path, repo_root: Path) -> Path:
        if spec.chemcpa_combined_h5ad and spec.chemcpa_combined_h5ad.exists():
            return spec.chemcpa_combined_h5ad
        combined = run_dir / "chemcpa_combined.h5ad"
        if combined.exists():
            return combined
        proc = run_python(
            repo_root,
            "scripts/baseline_exp/prepare_chemcpa_moa_h5ad.py",
            [
                "--train-path", str(spec.train_h5ad),
                "--test-path", str(spec.test_h5ad),
                "-o", str(combined),
            ],
            {},
        )
        ensure_success(proc, "ChemCPA h5ad preparation")
        return combined

    def train(self, spec: TaskSpec, run_dir: Path, repo_root: Path, env: dict, run_index: int) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        combined = self._combined_h5ad(spec, run_dir, repo_root)
        proc = run_python(
            repo_root,
            "scripts/baseline_exp/train_chemcpa_moa.py",
            [
                "--config", self.config,
                "--data-path", str(combined),
                "--save-dir", str(run_dir),
            ],
            env,
        )
        ensure_success(proc, f"{self.display_name} training")

    def evaluate(
        self,
        spec: TaskSpec,
        run_dir: Path,
        ckpt_path: Path,
        repo_root: Path,
        env: dict,
        run_index: int,
    ) -> Dict[str, float]:
        sample_dir = self.sample_run_dir(spec, run_index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        combined = self._combined_h5ad(spec, run_dir, repo_root)
        proc = run_python(
            repo_root,
            "scripts/baseline_exp/eval_chemcpa_moa.py",
            [
                "--config", self.config,
                "--ckpt", str(ckpt_path),
                "--data-path", str(combined),
                "--test-data-path", str(spec.test_h5ad),
                "--train-data-path", str(spec.train_h5ad),
                "--n_samples", str(spec.n_samples),
                "--out_h5ad", str(sample_dir / f"synthetic_{run_index}.h5ad"),
                "--drug-key", spec.drug_key,
                "--dose-key", spec.dose_key,
            ],
            env,
        )
        output = ensure_success(proc, f"{self.display_name} evaluation")
        return parse_metrics_from_output(output)
