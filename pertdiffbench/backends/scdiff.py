from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pertdiffbench.backends.base import PerturbationBackend, ensure_success, run_python
from pertdiffbench.evaluate import parse_metrics_from_output
from pertdiffbench.tasks.base import TaskSpec


class ScDiffBackend(PerturbationBackend):
    name = "scdiff"
    display_name = "scDiff"
    config = "configs/scdiff/eval_perturbation.yaml"

    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        return run_dir

    def _data_overrides(self, spec: TaskSpec, n_samples: int) -> List[str]:
        root = spec.scdiff_data_root or spec.train_h5ad.parent
        dataset = spec.scdiff_dataset or f"custom_{spec.subtask_id}"
        train_fname = spec.scdiff_train_fname or spec.train_h5ad.name
        test_fname = spec.scdiff_test_fname or spec.test_h5ad.name
        return [
            f"data.params.train.params.dataset={dataset}",
            f"data.params.train.params.fname={train_fname}",
            f"data.params.test.params.dataset={dataset}",
            f"data.params.test.params.fname={test_fname}",
            f"model.params.generation_kwargs.n_samples={n_samples}",
        ]

    def train(self, spec: TaskSpec, run_dir: Path, repo_root: Path, env: dict, run_index: int) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        spec: TaskSpec,
        run_dir: Path,
        ckpt_path: Path,
        repo_root: Path,
        env: dict,
        run_index: int,
    ) -> Dict[str, float]:
        data_root = spec.scdiff_data_root or spec.train_h5ad.parent
        logdir = spec.output_dir / "logs" / "scdiff"
        logdir.mkdir(parents=True, exist_ok=True)

        args = [
            "--custom_data_path", str(data_root),
            "--base", self.config,
            "--name", "v7.5",
            "--logdir", str(logdir),
            "--postfix", f"perturbation_v7.5_run{run_index}",
            "--model_save_path", str(run_dir),
            "--wandb_offline", "t",
            *self._data_overrides(spec, spec.n_samples),
        ]
        proc = run_python(repo_root, "src/scDiff/main.py", args, env)
        output = ensure_success(proc, f"{self.display_name} train+eval")
        return parse_metrics_from_output(output)
