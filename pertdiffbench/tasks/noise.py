from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pertdiffbench.tasks.base import BaseTask, TaskSpec

NOISE_TYPES = ("gaussian", "lognormal", "poisson", "zero_inflation")


class NoiseTask(BaseTask):
    """Robustness under synthetic noise on CD4T (train-once per noise level)."""

    name = "noise"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/noise",
        gene_nums: int = 6998,
        n_samples: int = 278,
        num_runs: int = 3,
        noise_type: str = "gaussian",
        noise_level: Optional[str] = None,
        **_,
    ) -> List[TaskSpec]:
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {NOISE_TYPES}")
        sub = f"{noise_type}_{noise_level}" if noise_level else noise_type
        return [
            TaskSpec(
                task_name=self.name,
                train_h5ad=train_h5ad,
                test_h5ad=test_h5ad,
                output_dir=Path(output_dir),
                gene_nums=gene_nums,
                n_samples=n_samples,
                num_runs=num_runs,
                subtask_id=sub,
                noise_type=noise_type,
                noise_level=noise_level,
                train_once=True,
            )
        ]
