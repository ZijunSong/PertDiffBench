from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pertdiffbench.backends.base import PerturbationBackend, ensure_success, run_python
from pertdiffbench.evaluate import parse_metrics_from_output
from pertdiffbench.tasks.base import TaskSpec


@dataclass(frozen=True)
class EncoderPipeline:
    apply_script: Optional[str]
    train_script: str
    eval_script: str
    config: str
    latent_key: str
    ckpt_name: str = "model_final.pth"
    ckpt_env: Optional[str] = None


ENCODER_PIPELINES: Dict[str, EncoderPipeline] = {
    "scgpt": EncoderPipeline(
        apply_script="scripts/encoder_exp/scgpt/apply_scgpt_encoder.py",
        train_script="scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py",
        eval_script="scripts/encoder_exp/scgpt/eval_latent_ddpm_mlp_generic.py",
        config="configs/baselines/scvi_ddpm_mlp.yaml",
        latent_key="X_scgpt",
        ckpt_env="SCGPT_CKPT_DIR",
    ),
    "scfoundation": EncoderPipeline(
        apply_script="scripts/encoder_exp/scfoundation/attach_scfoundation_embedding.py",
        train_script="scripts/encoder_exp/scfoundation/train_scfoundation_latent_ddpm_mlp.py",
        eval_script="scripts/encoder_exp/scfoundation/eval_scfoundation_latent_ddpm_mlp.py",
        config="configs/baselines/scfoundation_ddpm_mlp.yaml",
        latent_key="X_scfoundation",
        ckpt_env="SCFOUNDATION_CKPT_DIR",
    ),
    "geneformer": EncoderPipeline(
        apply_script="scripts/encoder_exp/geneformer/precompute_geneformer_latent.py",
        train_script="scripts/encoder_exp/geneformer/train_geneformer_latent_ddpm_mlp.py",
        eval_script="scripts/encoder_exp/geneformer/eval_geneformer_latent_ddpm_mlp.py",
        config="configs/baselines/geneformer_latent_ddpm_mlp.yaml",
        latent_key="X_geneformer",
        ckpt_env="GENEFORMER_DIR",
    ),
    "scvi": EncoderPipeline(
        apply_script="scripts/encoder_exp/scvi/apply_scvi_encoder.py",
        train_script="scripts/baseline_exp/train_scvi_ddpm_mlp.py",
        eval_script="scripts/baseline_exp/eval_scvi_ddpm_mlp.py",
        config="configs/baselines/scvi_ddpm_mlp.yaml",
        latent_key="X_scvi",
    ),
    "state": EncoderPipeline(
        apply_script="scripts/encoder_exp/state/apply_state_encoder.py",
        train_script="scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py",
        eval_script="scripts/encoder_exp/scgpt/eval_latent_ddpm_mlp_generic.py",
        config="configs/baselines/state_ddpm_mlp.yaml",
        latent_key="X_state",
        ckpt_env="STATE_MODEL_DIR",
    ),
    "scimilarity": EncoderPipeline(
        apply_script="scripts/encoder_exp/scimilarity/apply_scimilarity_encoder.py",
        train_script="scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py",
        eval_script="scripts/encoder_exp/scgpt/eval_latent_ddpm_mlp_generic.py",
        config="configs/baselines/scimilarity_ddpm_mlp.yaml",
        latent_key="X_scimilarity",
    ),
    "cellfm": EncoderPipeline(
        apply_script="scripts/encoder_exp/cellfm/apply_cellfm_encoder.py",
        train_script="scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py",
        eval_script="scripts/encoder_exp/scgpt/eval_latent_ddpm_mlp_generic.py",
        config="configs/baselines/cellfm_ddpm_mlp.yaml",
        latent_key="X_cellfm",
    ),
    "tahoe_x1": EncoderPipeline(
        apply_script="scripts/encoder_exp/tahoe-x1/apply_tx1_encoder.py",
        train_script="scripts/encoder_exp/scgpt/train_latent_ddpm_mlp_generic.py",
        eval_script="scripts/encoder_exp/scgpt/eval_latent_ddpm_mlp_generic.py",
        config="configs/baselines/encoder_tahoex1_ddpm.yaml",
        latent_key="X_tahoe_x1",
    ),
}


class EncoderBackend(PerturbationBackend):
    name = "encoder"
    display_name = "Encoder+DDPM"

    def supports(self, spec: TaskSpec) -> bool:
        return spec.task_name == "encoder" and spec.encoder_name in ENCODER_PIPELINES

    def display_name_for(self, spec: TaskSpec) -> str:
        return f"{spec.encoder_name}+DDPM"

    def _pipeline(self, spec: TaskSpec) -> EncoderPipeline:
        assert spec.encoder_name
        return ENCODER_PIPELINES[spec.encoder_name]

    def ckpt_path(self, spec: TaskSpec, run_dir: Path, run_index: int) -> Path:
        return run_dir / self._pipeline(spec).ckpt_name

    def _latent_paths(self, spec: TaskSpec) -> tuple[Path, Path]:
        cache = spec.output_dir / "latent_cache"
        cache.mkdir(parents=True, exist_ok=True)
        name = spec.encoder_name
        return cache / f"train_{name}_latent.h5ad", cache / f"test_{name}_latent.h5ad"

    def _encode_if_needed(
        self, spec: TaskSpec, repo_root: Path, env: dict, pipe: EncoderPipeline
    ) -> tuple[Path, Path]:
        train_lat, test_lat = self._latent_paths(spec)
        if train_lat.exists() and test_lat.exists():
            return train_lat, test_lat

        if not pipe.apply_script:
            return spec.train_h5ad, spec.test_h5ad

        ckpt_dir = spec.encoder_ckpt_dir
        if ckpt_dir is None and pipe.ckpt_env and pipe.ckpt_env in os.environ:
            ckpt_dir = Path(os.environ[pipe.ckpt_env])

        for src, dst in ((spec.train_h5ad, train_lat), (spec.test_h5ad, test_lat)):
            if dst.exists():
                continue
            args = ["--input", str(src), "--output", str(dst)]
            if ckpt_dir:
                args.extend(["--ckpt-dir", str(ckpt_dir)])
            proc = run_python(repo_root, pipe.apply_script, args, env)
            ensure_success(proc, f"{spec.encoder_name} encoding {dst.name}")

        if train_lat.exists() and test_lat.exists():
            return train_lat, test_lat
        return spec.train_h5ad, spec.test_h5ad

    def train(self, spec: TaskSpec, run_dir: Path, repo_root: Path, env: dict, run_index: int) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        pipe = self._pipeline(spec)
        train_data, _ = self._encode_if_needed(spec, repo_root, env, pipe)
        latent_key = spec.latent_key or pipe.latent_key

        if pipe.train_script.endswith("train_scvi_ddpm_mlp.py"):
            proc = run_python(
                repo_root, pipe.train_script,
                ["-c", pipe.config],
                env,
            )
        elif "train_geneformer" in pipe.train_script or "train_scfoundation" in pipe.train_script:
            proc = run_python(
                repo_root,
                pipe.train_script,
                [
                    "-c", pipe.config,
                    "--train-data-path", str(train_data),
                    "--save-weight-dir", str(run_dir),
                ],
                env,
            )
        else:
            proc = run_python(
                repo_root,
                pipe.train_script,
                [
                    "-c", pipe.config,
                    "--train-data-path", str(train_data),
                    "--latent-key", latent_key,
                    "--save-weight-dir", str(run_dir),
                ],
                env,
            )
        ensure_success(proc, f"{self.display_name_for(spec)} training")

    def evaluate(
        self,
        spec: TaskSpec,
        run_dir: Path,
        ckpt_path: Path,
        repo_root: Path,
        env: dict,
        run_index: int,
    ) -> Dict[str, float]:
        pipe = self._pipeline(spec)
        _, test_data = self._encode_if_needed(spec, repo_root, env, pipe)
        latent_key = spec.latent_key or pipe.latent_key
        sample_dir = self.sample_run_dir(spec, run_index)
        sample_dir.mkdir(parents=True, exist_ok=True)

        if "eval_scvi" in pipe.eval_script or "eval_scfoundation" in pipe.eval_script or "eval_geneformer" in pipe.eval_script:
            args = [
                "-c", pipe.config,
                "-k", str(ckpt_path),
                "-n", str(spec.n_samples),
                "-o", str(sample_dir / f"preds_run_{run_index}.h5ad"),
            ]
            if "eval_scfoundation" in pipe.eval_script or "eval_geneformer" in pipe.eval_script:
                args.extend(["--data-path", str(test_data)])
        else:
            args = [
                "-c", pipe.config,
                "-k", str(ckpt_path),
                "--data-path", str(test_data),
                "--latent-key", latent_key,
                "-n", str(spec.n_samples),
                "-o", str(sample_dir / f"preds_run_{run_index}.h5ad"),
            ]

        proc = run_python(repo_root, pipe.eval_script, args, env)
        output = ensure_success(proc, f"{self.display_name_for(spec)} evaluation")
        return parse_metrics_from_output(output)
