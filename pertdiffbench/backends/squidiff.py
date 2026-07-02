from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pertdiffbench.backends.base import (
    MOA_TASKS,
    TEMPORAL_TASKS,
    PerturbationBackend,
    ensure_success,
    run_env_for_index,
    run_python,
)
from pertdiffbench.evaluate import parse_metrics_from_output
from pertdiffbench.tasks.base import TaskSpec


class SquidiffBackend(PerturbationBackend):
    name = "squidiff"
    display_name = "Squidiff"
    ckpt_name = "model.pt"

    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        if self.squidiff_train_once(spec):
            return spec.ckpt_dir / self.name / self.ckpt_name
        return run_dir / self.ckpt_name

    def ckpt_root(self, spec: TaskSpec, run_index: int) -> Path:
        if self.squidiff_train_once(spec):
            return spec.ckpt_dir / self.name
        return self.run_dir(spec, run_index)

    def _train_args(self, spec: TaskSpec, ckpt_root: Path, repo_root: Path) -> List[str]:
        log_dir = spec.output_dir / "logs" / self.name / spec.subtask_id
        log_dir.mkdir(parents=True, exist_ok=True)
        data = self.training_data(spec)
        args = [
            "--logger_path", str(log_dir),
            "--data_path", str(data),
            "--resume_checkpoint", str(ckpt_root),
            "--gene_size", str(spec.gene_nums),
            "--output_dim", str(spec.gene_nums),
        ]
        if spec.task_name in MOA_TASKS and spec.use_drug_structure:
            args.extend(["--use_drug_structure", "True"])
        return args

    def train(self, spec: TaskSpec, run_dir: Path, repo_root: Path, env: dict, run_index: int) -> None:
        ckpt_root = self.ckpt_root(spec, run_index)
        ckpt_root.mkdir(parents=True, exist_ok=True)
        proc = run_python(
            repo_root,
            "src/Squidiff/train_squidiff.py",
            self._train_args(spec, ckpt_root, repo_root),
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

        if spec.task_name in TEMPORAL_TASKS:
            synthetic = sample_dir / "synthetic_fig4.h5ad"
            proc_s = run_python(
                repo_root,
                "scripts/fig4/sample_fig4_squidiff_interp.py",
                [
                    "--model_path", str(ckpt_path),
                    "--train-h5ad", str(spec.train_h5ad),
                    "--out-h5ad", str(synthetic),
                    "--n-samples", str(spec.n_samples),
                    "--gene-size", str(spec.gene_nums),
                    "--output-dim", str(spec.gene_nums),
                    "--method", "addition",
                    "--anchor-start", "2h",
                    "--anchor-end", "8h",
                    "--target-times", "4h", "6h",
                ],
                env,
            )
            ensure_success(proc_s, f"{self.display_name} temporal sampling")
            proc = run_python(
                repo_root,
                "scripts/fig4/eval_fig4_time_conditioned.py",
                [
                    "--test-h5ad", str(spec.test_h5ad),
                    "--generated-h5ad", str(synthetic),
                    "--train-h5ad", str(spec.train_h5ad),
                    "--n-samples", str(spec.n_samples),
                ],
                env,
            )
        else:
            args = [
                "--model_path", str(ckpt_path),
                "--gene_size", str(spec.gene_nums),
                "--output_dim", str(spec.gene_nums),
                "--out_h5ad", str(sample_dir / f"synthetic_ifn_run_{run_index}.h5ad"),
                "--train_data_path", str(spec.test_h5ad),
                "--n_samples", str(spec.n_samples),
                "--umap_plot", str(sample_dir / f"umap_comparison_{run_index}.png"),
                "--data_path", str(spec.test_h5ad),
            ]
            if spec.task_name in MOA_TASKS and spec.use_drug_structure:
                args.extend(["--use_drug_structure", "True"])
            proc = run_python(repo_root, "src/Squidiff/sample_squidiff.py", args, env)

        output = ensure_success(proc, f"{self.display_name} evaluation")
        return parse_metrics_from_output(output)

    def run_all_evals(
        self, spec: TaskSpec, repo_root: Path, env: dict
    ) -> List[Dict[str, float]]:
        ckpt_path = self.ckpt_path(spec, spec.ckpt_dir / self.name, 1)
        if not self.squidiff_train_once(spec):
            results = []
            for run_index in range(1, spec.num_runs + 1):
                run_dir = self.run_dir(spec, run_index)
                run_env = run_env_for_index(env, run_index)
                self.train(spec, run_dir, repo_root, run_env, run_index)
                ckpt = self.ckpt_path(spec, run_dir, run_index)
                results.append(self.evaluate(spec, run_dir, ckpt, repo_root, run_env, run_index))
            return results

        ckpt_root = spec.ckpt_dir / self.name
        self.train(spec, ckpt_root, repo_root, run_env_for_index(env, 1), 1)
        results = []
        for run_index in range(1, spec.num_runs + 1):
            results.append(
                self.evaluate(
                    spec, ckpt_root, ckpt_path, repo_root,
                    run_env_for_index(env, run_index), run_index,
                )
            )
        return results
