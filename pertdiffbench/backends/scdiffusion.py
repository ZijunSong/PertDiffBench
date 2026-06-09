from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from pertdiffbench.backends.base import (
    TEMPORAL_TASKS,
    PerturbationBackend,
    ensure_success,
    run_python,
)
from pertdiffbench.evaluate import parse_metrics_from_output
from pertdiffbench.tasks.base import TaskSpec


class ScDiffusionBackend(PerturbationBackend):
    name = "scdiffusion"
    display_name = "scDiffusion"

    def supports(self, spec: TaskSpec) -> bool:
        if not super().supports(spec):
            return False
        return spec.task_name != "encoder"

    def _annotation_dir(self, spec: TaskSpec) -> Path:
        if spec.annotation_model_dir:
            return spec.annotation_model_dir
        return Path(
            os.environ.get(
                "ANNOTATION_MODEL_DIR",
                "/data/ppnm/checkpoints/PertDiffBench/checkpoints/annotation_model_v1",
            )
        )

    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        return run_dir / "diffusion" / "my_diffusion" / "model010000.pt"

    def _paths(self, spec: TaskSpec, run_index: int) -> dict:
        run_dir = self.run_dir(spec, run_index)
        vae_dir = run_dir / "vae"
        diff_dir = run_dir / "diffusion"
        cls_dir = run_dir / "classifier"
        for d in (vae_dir, diff_dir, cls_dir):
            d.mkdir(parents=True, exist_ok=True)
        return {
            "run_dir": run_dir,
            "vae_dir": vae_dir,
            "diff_dir": diff_dir,
            "cls_dir": cls_dir,
            "vae_ckpt": vae_dir / "model_seed=0_step=9999.pt",
            "diff_ckpt": diff_dir / "my_diffusion" / "model010000.pt",
            "cls_ckpt": cls_dir / "model009999.pt",
        }

    def train(self, spec: TaskSpec, run_dir: Path, repo_root: Path, env: dict, run_index: int) -> None:
        paths = self._paths(spec, run_index)
        data = str(self.training_data(spec))
        ann = str(self._annotation_dir(spec))

        proc_vae = run_python(
            repo_root,
            "src/scDiffusion/VAE/VAE_train.py",
            [
                "--data_dir", data,
                "--num_genes", str(spec.gene_nums),
                "--state_dict", ann,
                "--save_dir", str(paths["vae_dir"]),
            ],
            env,
            cwd=repo_root / "src/scDiffusion/VAE",
        )
        ensure_success(proc_vae, f"{self.display_name} VAE training")

        proc_diff = run_python(
            repo_root,
            "src/scDiffusion/cell_train.py",
            [
                "--data_dir", data,
                "--vae_path", str(paths["vae_ckpt"]),
                "--save_dir", str(paths["diff_dir"]),
            ],
            env,
            cwd=repo_root / "src/scDiffusion",
        )
        ensure_success(proc_diff, f"{self.display_name} diffusion training")

        proc_cls = run_python(
            repo_root,
            "src/scDiffusion/classifier_train.py",
            [
                "--data_dir", data,
                "--vae_path", str(paths["vae_ckpt"]),
                "--model_path", str(paths["cls_dir"]),
            ],
            env,
            cwd=repo_root / "src/scDiffusion",
        )
        ensure_success(proc_cls, f"{self.display_name} classifier training")

    def evaluate(
        self,
        spec: TaskSpec,
        run_dir: Path,
        ckpt_path: Path,
        repo_root: Path,
        env: dict,
        run_index: int,
    ) -> Dict[str, float]:
        vae_dir = run_dir / "vae"
        diff_dir = run_dir / "diffusion"
        cls_dir = run_dir / "classifier"
        paths = {
            "vae_ckpt": vae_dir / "model_seed=0_step=9999.pt",
            "diff_ckpt": diff_dir / "my_diffusion" / "model010000.pt",
            "cls_ckpt": cls_dir / "model009999.pt",
        }
        sample_dir = self.sample_run_dir(spec, run_index)
        sample_dir.mkdir(parents=True, exist_ok=True)
        data = str(self.training_data(spec))

        if spec.task_name in TEMPORAL_TASKS:
            sample_script = "classifier_sample_pre.py"
        else:
            sample_script = "classifier_sample.py"

        proc = run_python(
            repo_root,
            f"src/scDiffusion/{sample_script}",
            [
                "--num_samples", str(spec.n_samples),
                "--train-data-path", data,
                "--model_path", str(paths["diff_ckpt"]),
                "--classifier_path", str(paths["cls_ckpt"]),
                "--ae_dir", str(paths["vae_ckpt"]),
                "--num_gene", str(spec.gene_nums),
                "--sample_dir", str(sample_dir),
                "--out_h5ad", str(sample_dir / f"synthetic_ifn_{run_index}.h5ad"),
                "--umap_plot", str(sample_dir / f"umap_comparison_{run_index}.png"),
                "--init_cell_path", str(spec.test_h5ad),
            ],
            env,
            cwd=repo_root / "src/scDiffusion",
        )
        output = ensure_success(proc, f"{self.display_name} sampling/evaluation")
        return parse_metrics_from_output(output)

    def train_once_for_plus(self, spec: TaskSpec) -> bool:
        return spec.task_name == "cross_celltype_plus"
