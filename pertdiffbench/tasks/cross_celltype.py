from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pertdiffbench.tasks.base import BaseTask, TaskSpec


class CrossCelltypeTask(BaseTask):
    """Train on CD4T, evaluate on unseen cell types (strict OOD, no test-domain training data)."""

    name = "cross_celltype"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/cross_celltype",
        gene_nums: int = 6998,
        n_samples: int = 0,
        num_runs: int = 3,
        celltype_to_predict: Optional[str] = None,
        **_,
    ) -> List[TaskSpec]:
        sub = celltype_to_predict or Path(test_h5ad).stem
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
                celltype_to_predict=celltype_to_predict,
            )
        ]


class CrossCelltypeExtendTask(BaseTask):
    """scGen-style: train on CD4T + test cell type controls (pair split=train)."""

    name = "cross_celltype_extend"

    def build_specs(
        self,
        combined_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/cross_celltype_extend",
        gene_nums: int = 6998,
        n_samples: int = 0,
        num_runs: int = 3,
        celltype_to_predict: Optional[str] = None,
        train_h5ad: Optional[str | Path] = None,
        **_,
    ) -> List[TaskSpec]:
        sub = celltype_to_predict or Path(test_h5ad).stem
        return [
            TaskSpec(
                task_name=self.name,
                train_h5ad=train_h5ad or combined_h5ad,
                test_h5ad=test_h5ad,
                combined_h5ad=Path(combined_h5ad),
                output_dir=Path(output_dir),
                gene_nums=gene_nums,
                n_samples=n_samples,
                num_runs=num_runs,
                subtask_id=sub,
                celltype_to_predict=celltype_to_predict,
                pair_only_obs_key="split",
                pair_only_obs_value="train",
            )
        ]


class CrossCelltypePlusTask(BaseTask):
    """Leave-one-out cell type with partial held-out controls (p0 / p0.25 / p0.5)."""

    name = "cross_celltype_plus"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        combined_h5ad: str | Path,
        output_dir: str | Path = "runs/cross_celltype_plus",
        gene_nums: int = 1000,
        n_samples: int = 0,
        num_runs: int = 3,
        held_out_celltype: str = "B",
        control_fraction: str = "p0.25",
        **_,
    ) -> List[TaskSpec]:
        sub = f"loo_{held_out_celltype}_{control_fraction}"
        return [
            TaskSpec(
                task_name=self.name,
                train_h5ad=train_h5ad,
                test_h5ad=test_h5ad,
                combined_h5ad=Path(combined_h5ad),
                output_dir=Path(output_dir),
                gene_nums=gene_nums,
                n_samples=n_samples,
                num_runs=num_runs,
                subtask_id=sub,
                celltype_to_predict=held_out_celltype,
                pair_only_obs_key="split",
                pair_only_obs_value="train",
                train_once=False,
            )
        ]
