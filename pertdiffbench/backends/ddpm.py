from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pertdiffbench.backends.base import (
    MOA_TASKS,
    TEMPORAL_TASKS,
    PerturbationBackend,
    ensure_success,
    pair_only_args,
    run_python,
)
from pertdiffbench.evaluate import parse_metrics_from_output
from pertdiffbench.tasks.base import TaskSpec


class DDPBackend(PerturbationBackend):
    name = "ddpm"
    display_name = "DDPM"
    config = "configs/baselines/scrna_ddpm_scrna.yaml"
    ckpt_name = "scrna_ddpm_epoch1000.pt"
    ae_config = "configs/baselines/mlp_ddpm_mlp.yaml"
    ae_ckpt_name = "ae_epoch_1000.pth"

    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        return run_dir / self.ckpt_name

    def _is_moa(self, spec: TaskSpec) -> bool:
        return spec.task_name in MOA_TASKS

    def train(self, spec: TaskSpec, run_dir: Path, repo_root: Path, env: dict, run_index: int) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        data = self.training_data(spec)

        if spec.task_name in TEMPORAL_TASKS:
            ae_dir = spec.ckpt_dir / "fig4_ae" / f"run{run_index}"
            ae_dir.mkdir(parents=True, exist_ok=True)
            proc_ae = run_python(
                repo_root,
                "scripts/fig4/train_fig4_ae_for_ddpm.py",
                [
                    "--config", self.ae_config,
                    "--data-path", str(spec.train_h5ad),
                    "--save-dir", str(ae_dir),
                    "--gene-nums", str(spec.gene_nums),
                ],
                env,
            )
            ensure_success(proc_ae, f"{self.display_name} Fig4 AE training")

        if self._is_moa(spec):
            args = [
                "--config", self.config,
                "--data-path", str(data),
                "--save-weight-dir", str(run_dir),
                "--gene-nums", str(spec.gene_nums),
                "--drug-key", spec.drug_key,
                "--dose-key", spec.dose_key,
            ]
            if spec.use_drug_structure:
                args.append("--use-drug-structure")
            proc = run_python(repo_root, "scripts/baseline_exp/train_scrna_ddpm_scrna_moa.py", args, env)
        else:
            args = [
                "--config", self.config,
                "--data-path", str(data),
                "--save-weight-dir", str(run_dir),
                "--gene-nums", str(spec.gene_nums),
                *pair_only_args(spec),
            ]
            proc = run_python(repo_root, "scripts/baseline_exp/train_scrna_ddpm_scrna.py", args, env)
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
        synthetic = sample_dir / f"synthetic_ifn_run{run_index}.h5ad"

        if spec.task_name in TEMPORAL_TASKS:
            ae_ckpt = spec.ckpt_dir / "fig4_ae" / f"run{run_index}" / self.ae_ckpt_name
            synthetic = sample_dir / "synthetic_fig4.h5ad"
            if ae_ckpt.exists():
                proc_s = run_python(
                    repo_root,
                    "scripts/fig4/sample_fig4_vae_linear_interp.py",
                    [
                        "--config", self.ae_config,
                        "--ckpt", str(ae_ckpt),
                        "--train-h5ad", str(spec.train_h5ad),
                        "--out-h5ad", str(synthetic),
                        "--n-samples", str(spec.n_samples),
                        "--gene-nums", str(spec.gene_nums),
                    ],
                    env,
                )
                ensure_success(proc_s, f"{self.display_name} temporal sampling")

            args = [
                "--config", self.config,
                "--train-data-path", str(spec.train_h5ad),
                "--data-path", str(spec.test_h5ad),
                "--ckpt", str(ckpt_path),
                "--time-conditioned",
                "--generated-h5ad", str(synthetic),
                "--n_samples", str(spec.n_samples),
                "--gene-nums", str(spec.gene_nums),
            ]
            proc = run_python(repo_root, "scripts/baseline_exp/eval_scrna_ddpm_scrna.py", args, env)
        elif self._is_moa(spec):
            label_enc = run_dir / "label_encoder.npz"
            args = [
                "--config", self.config,
                "--ckpt", str(ckpt_path),
                "--label-encoder-path", str(label_enc),
                "--data-path", str(spec.test_h5ad),
                "--train-data-path", str(spec.train_h5ad),
                "--n_samples", str(spec.n_samples),
                "--out_h5ad", str(synthetic),
                "--gene-nums", str(spec.gene_nums),
                "--drug-key", spec.drug_key,
                "--dose-key", spec.dose_key,
            ]
            if spec.use_drug_structure:
                args.append("--use-drug-structure")
            if not spec.skip_umap:
                args.extend(["--umap_plot", str(sample_dir / f"umap_{run_index}.png")])
            proc = run_python(repo_root, "scripts/baseline_exp/eval_scrna_ddpm_scrna_moa.py", args, env)
        else:
            args = [
                "--config", self.config,
                "--train-data-path", str(spec.train_h5ad),
                "--data-path", str(spec.test_h5ad),
                "--ckpt", str(ckpt_path),
                "--out_h5ad", str(synthetic),
                "--n_samples", str(spec.n_samples),
                "--gene-nums", str(spec.gene_nums),
                "--umap_plot", str(sample_dir / f"umap_comparison_{run_index}.png"),
            ]
            proc = run_python(repo_root, "scripts/baseline_exp/eval_scrna_ddpm_scrna.py", args, env)

        output = ensure_success(proc, f"{self.display_name} evaluation")
        return parse_metrics_from_output(output)
