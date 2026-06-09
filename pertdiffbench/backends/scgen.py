from __future__ import annotations

from pathlib import Path
from typing import Dict

from pertdiffbench.backends.base import PerturbationBackend, ensure_success, run_python
from pertdiffbench.evaluate import parse_metrics_from_output
from pertdiffbench.tasks.base import TaskSpec


class ScGenBackend(PerturbationBackend):
    name = "scgen"
    display_name = "scGen"

    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        return run_dir / "model.pt"

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
        sample_dir = self.sample_run_dir(spec, run_index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        model_dir = run_dir

        celltype = spec.celltype_to_predict or "CD4T"
        if spec.species_to_predict:
            celltype = spec.species_to_predict

        train_path = self.training_data(spec)
        proc = run_python(
            repo_root,
            "scripts/scGen_eval.py",
            [
                "--train_data_path", str(train_path),
                "--test_data_path", str(spec.test_h5ad),
                "--model_save_path", str(model_dir),
                "--celltype_to_predict", str(celltype),
                "--out_h5ad", str(sample_dir / f"synthetic_run_{run_index}.h5ad"),
                "--umap_plot", str(sample_dir / f"umap_{run_index}.png"),
                "--n_samples", str(spec.n_samples),
            ],
            env,
        )
        output = ensure_success(proc, f"{self.display_name} train+eval")
        return parse_metrics_from_output(output)
