from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pertdiffbench.tasks.base import BaseTask, TaskSpec


class CrossSpeciesTask(BaseTask):
    """Train on mouse, evaluate on pig/rabbit/rat (train-once on source species)."""

    name = "cross_species"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/cross_species",
        gene_nums: int = 6619,
        n_samples: int = 1000,
        num_runs: int = 3,
        species_to_predict: Optional[str] = None,
        **_,
    ) -> List[TaskSpec]:
        sub = species_to_predict or Path(test_h5ad).stem.replace("_control_ifn", "")
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
                species_to_predict=species_to_predict or sub,
                celltype_to_predict=species_to_predict or sub,
                train_once=True,
            )
        ]


class CrossSpeciesLooTask(BaseTask):
    name = "cross_species_loo"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/cross_species_loo",
        gene_nums: int = 6619,
        n_samples: int = 1000,
        num_runs: int = 3,
        held_out_species: str = "pig",
        **_,
    ) -> List[TaskSpec]:
        return CrossSpeciesTask().build_specs(
            train_h5ad=train_h5ad,
            test_h5ad=test_h5ad,
            output_dir=output_dir,
            gene_nums=gene_nums,
            n_samples=n_samples,
            num_runs=num_runs,
            species_to_predict=held_out_species,
        )
