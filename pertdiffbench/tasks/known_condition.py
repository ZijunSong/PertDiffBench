from __future__ import annotations

from pathlib import Path
from typing import List

from pertdiffbench.tasks.base import BaseTask, TaskSpec


class KnownConditionTask(BaseTask):
    name = "known_condition"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/known_condition",
        gene_nums: int = 1000,
        n_samples: int = 100,
        num_runs: int = 3,
        squidiff_train_once: bool = True,
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
                squidiff_train_once=squidiff_train_once,
            )
        ]
