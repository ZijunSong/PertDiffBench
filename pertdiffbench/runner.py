from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from pertdiffbench.backends.base import PerturbationBackend, run_env_for_index
from pertdiffbench.backends.encoder import EncoderBackend
from pertdiffbench.backends.scdiffusion import ScDiffusionBackend
from pertdiffbench.backends.squidiff import SquidiffBackend
from pertdiffbench.evaluate import aggregate_runs, save_metrics_csv
from pertdiffbench.registry import MODEL_REGISTRY, SUPPORTED_MODELS
from pertdiffbench.tasks.base import BaseTask, TaskSpec
from pertdiffbench.tasks.registry import TASK_REGISTRY, SUPPORTED_TASKS


@dataclass
class RunResult:
    method: str
    subtask_id: str
    metrics_df: pd.DataFrame
    csv_path: Path


@dataclass
class BenchmarkRunner:
    """Run multiple backends on one or more perturbation-prediction task specs."""

    task: Union[str, BaseTask] = "known_condition"
    output_dir: Union[str, Path] = "runs/default"
    repo_root: Optional[Path] = None
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    _task_impl: BaseTask = field(init=False, repr=False)
    _specs: List[TaskSpec] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.repo_root is None:
            self.repo_root = self._detect_repo_root()
        else:
            self.repo_root = Path(self.repo_root)
        self.output_dir = Path(self.output_dir)

        if isinstance(self.task, str):
            if self.task not in TASK_REGISTRY:
                raise ValueError(f"Unknown task: {self.task}. Available: {SUPPORTED_TASKS}")
            self._task_impl = TASK_REGISTRY[self.task]
        else:
            self._task_impl = self.task

    @staticmethod
    def _detect_repo_root() -> Path:
        here = Path(__file__).resolve().parent.parent
        if (here / "scripts" / "baseline_exp").exists():
            return here
        raise FileNotFoundError(
            "Could not detect PertDiffBench repo root. Pass repo_root explicitly."
        )

    def build_specs(self, **kwargs) -> List[TaskSpec]:
        merged = {**self.task_kwargs, **kwargs}
        merged.setdefault("output_dir", self.output_dir)
        specs = self._task_impl.build_specs(**merged)
        for spec in specs:
            if spec.n_samples is None or spec.n_samples <= 0:
                from utils.max_eval_samples import resolve_eval_n_samples

                mode_by_task = {
                    "moa_same": "multi_pert",
                    "moa_diff": "multi_pert",
                    "cross_celltype": "multi_pert",
                    "cross_celltype_extend": "multi_pert",
                    "cross_celltype_plus": "multi_pert",
                    "cross_species": "timepoint",
                    "cross_species_loo": "timepoint",
                    "temporal": "timepoint",
                    "encoder": "multi_pert",
                    "noise": "paired_ifn",
                }
                mode = mode_by_task.get(spec.task_name, "paired_ifn")
                spec.n_samples = resolve_eval_n_samples(spec.test_h5ad, 0, mode=mode)
            self._task_impl.validate(spec)
        self._specs = specs
        return specs

    def _runtime_env(self) -> dict:
        env = os.environ.copy()
        root = str(self.repo_root)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _method_label(self, backend: PerturbationBackend, spec: TaskSpec) -> str:
        if isinstance(backend, EncoderBackend) and spec.encoder_name:
            return backend.display_name_for(spec)
        return backend.display_name

    def _run_backend_on_spec(
        self,
        backend: PerturbationBackend,
        spec: TaskSpec,
        env: dict,
        skip_train: bool,
    ) -> List[Dict[str, float]]:
        if not backend.supports(spec):
            print(f"  Skipping {backend.name}: not supported for task {spec.task_name}")
            return []

        if isinstance(backend, SquidiffBackend) and backend.squidiff_train_once(spec):
            if skip_train and backend.ckpt_path(spec, spec.ckpt_dir / backend.name, 1).exists():
                metrics = []
                ckpt = backend.ckpt_path(spec, spec.ckpt_dir / backend.name, 1)
                for run_index in range(1, spec.num_runs + 1):
                    metrics.append(
                        backend.evaluate(
                            spec, spec.ckpt_dir / backend.name, ckpt,
                            self.repo_root, run_env_for_index(env, run_index), run_index,
                        )
                    )
                return metrics
            return backend.run_all_evals(spec, self.repo_root, env)

        if isinstance(backend, ScDiffusionBackend) and backend.train_once_for_plus(spec):
            run_dir = backend.run_dir(spec, 1)
            ckpt = backend.ckpt_path(spec, run_dir, 1)
            if not skip_train or not ckpt.exists():
                backend.train(spec, run_dir, self.repo_root, run_env_for_index(env, 1), 1)
            metrics = []
            for run_index in range(1, spec.num_runs + 1):
                metrics.append(
                    backend.evaluate(
                        spec, run_dir, ckpt, self.repo_root,
                        run_env_for_index(env, run_index), run_index,
                    )
                )
            return metrics

        if backend.train_once(spec):
            run_dir = backend.run_dir(spec, 1)
            ckpt = backend.ckpt_path(spec, run_dir, 1)
            if not skip_train or not ckpt.exists():
                print(f"  [{backend.display_name}] Training once for {spec.subtask_id}")
                backend.train(spec, run_dir, self.repo_root, run_env_for_index(env, 1), 1)
            metrics = []
            for run_index in range(1, spec.num_runs + 1):
                print(f"  [{backend.display_name}] Evaluating run {run_index}/{spec.num_runs}")
                metrics.append(
                    backend.evaluate(
                        spec, run_dir, ckpt, self.repo_root,
                        run_env_for_index(env, run_index), run_index,
                    )
                )
            return metrics

        metrics: List[Dict[str, float]] = []
        for run_index in range(1, spec.num_runs + 1):
            run_dir = backend.run_dir(spec, run_index)
            ckpt = backend.ckpt_path(spec, run_dir, run_index)
            run_env = run_env_for_index(env, run_index)
            if not skip_train or not ckpt.exists():
                print(f"  [{backend.display_name}] Training run {run_index}/{spec.num_runs}")
                backend.train(spec, run_dir, self.repo_root, run_env, run_index)
            else:
                print(f"  [{backend.display_name}] Skipping training run {run_index}")
            print(f"  [{backend.display_name}] Evaluating run {run_index}/{spec.num_runs}")
            metrics.append(
                backend.evaluate(spec, run_dir, ckpt, self.repo_root, run_env, run_index)
            )
        return metrics

    def run(
        self,
        models: Optional[List[str]] = None,
        skip_train: bool = False,
        **build_kwargs: Any,
    ) -> pd.DataFrame:
        specs = self._specs or self.build_specs(**build_kwargs)
        env = self._runtime_env()
        selected = models or [m for m in SUPPORTED_MODELS if m != "encoder"]

        if self._task_impl.name == "encoder":
            selected = ["encoder"]

        unknown = [m for m in selected if m not in MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}. Available: {SUPPORTED_MODELS}")

        all_frames: List[pd.DataFrame] = []

        for spec in specs:
            print(f"\n{'#' * 70}\nTask spec: {spec.task_name} / {spec.subtask_id}\n{'#' * 70}")
            spec_frames: List[pd.DataFrame] = []

            for model_name in selected:
                backend = MODEL_REGISTRY[model_name]
                label = self._method_label(backend, spec)
                print(f"\n{'=' * 60}\nRunning {label} ({model_name})\n{'=' * 60}")

                run_metrics = self._run_backend_on_spec(backend, spec, env, skip_train)
                if not run_metrics:
                    continue

                df = aggregate_runs(label, run_metrics)
                csv_path = spec.output_dir / f"metrics_{model_name}.csv"
                save_metrics_csv(df, str(csv_path))
                spec_frames.append(df)
                all_frames.append(df)
                print(f"Saved metrics to {csv_path}")

            if spec_frames:
                merged_spec = pd.concat(spec_frames, ignore_index=True)
                merged_path = spec.output_dir / "metrics_all_models.csv"
                save_metrics_csv(merged_spec, str(merged_path))

        merged = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        if len(specs) == 1:
            global_path = specs[0].output_dir / "metrics_all_models.csv"
        else:
            global_path = self.output_dir / "metrics_all_models.csv"
            global_path.parent.mkdir(parents=True, exist_ok=True)
        save_metrics_csv(merged, str(global_path))
        print(f"\nMerged metrics saved to {global_path}")
        return merged

    def list_models(self, task: Optional[str] = None) -> List[str]:
        task_name = task or (self.task if isinstance(self.task, str) else self._task_impl.name)
        dummy_kwargs = {
            "train_h5ad": "/tmp/train.h5ad",
            "test_h5ad": "/tmp/test.h5ad",
            "data_root": "/tmp",
            "encoder_name": "scgpt",
        }
        try:
            task_impl = TASK_REGISTRY[task_name]
            specs = task_impl.build_specs(**dummy_kwargs)
            spec = specs[0]
        except Exception:
            spec = TaskSpec(
                task_name=task_name,
                train_h5ad=Path("/tmp/train.h5ad"),
                test_h5ad=Path("/tmp/test.h5ad"),
                output_dir=Path("/tmp"),
            )
        return [m for m, b in MODEL_REGISTRY.items() if b.supports(spec)]

    def list_tasks(self) -> List[str]:
        return list(SUPPORTED_TASKS)
