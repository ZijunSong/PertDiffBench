from __future__ import annotations

from pathlib import Path
from typing import List

from pertdiffbench.tasks.base import BaseTask, TaskSpec


class TemporalTask(BaseTask):
    """A549 dexamethasone time imputation: train 0/2/8/10h, test 4/6h."""

    name = "temporal"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/temporal",
        gene_nums: int = 3000,
        n_samples: int = 500,
        num_runs: int = 3,
        **_,
    ) -> List[TaskSpec]:
        return [
            TaskSpec(
                task_name=self.name,
                train_h5ad=train_h5ad,
                test_h5ad=test_h5ad,
                output_dir=Path(output_dir),
                gene_nums=gene_nums,
                n_samples=n_samples,
                num_runs=num_runs,
            )
        ]
