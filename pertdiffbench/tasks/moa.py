from __future__ import annotations

from pathlib import Path
from typing import List

from pertdiffbench.tasks.base import BaseTask, TaskSpec


def _discover_moa_pairs(data_root: Path) -> List[tuple[Path, Path, str]]:
    pairs: List[tuple[Path, Path, str]] = []
    for train_path in sorted(data_root.glob("*_train__plus_control.h5ad")):
        moa = train_path.name.replace("_train__plus_control.h5ad", "")
        test_path = data_root / f"{moa}_test__plus_control.h5ad"
        if test_path.exists():
            pairs.append((train_path, test_path, moa))
    return pairs


class MoaSameTask(BaseTask):
    name = "moa_same"

    def build_specs(
        self,
        data_root: str | Path,
        output_dir: str | Path = "runs/moa_same",
        gene_nums: int = 3000,
        n_samples: int = 0,
        num_runs: int = 3,
        use_drug_structure: bool = True,
        moa_name: str | None = None,
        train_h5ad: str | Path | None = None,
        test_h5ad: str | Path | None = None,
        **_,
    ) -> List[TaskSpec]:
        root = Path(data_root)
        specs: List[TaskSpec] = []

        if train_h5ad and test_h5ad:
            moa = moa_name or Path(train_h5ad).stem.replace("_train__plus_control", "")
            specs.append(
                TaskSpec(
                    task_name=self.name,
                    train_h5ad=train_h5ad,
                    test_h5ad=test_h5ad,
                    output_dir=Path(output_dir),
                    gene_nums=gene_nums,
                    n_samples=n_samples,
                    num_runs=num_runs,
                    subtask_id=moa,
                    moa_split="same",
                    moa_name=moa,
                    use_drug_structure=use_drug_structure,
                )
            )
            return specs

        for train_path, test_path, moa in _discover_moa_pairs(root):
            if moa_name and moa != moa_name:
                continue
            specs.append(
                TaskSpec(
                    task_name=self.name,
                    train_h5ad=train_path,
                    test_h5ad=test_path,
                    output_dir=Path(output_dir),
                    gene_nums=gene_nums,
                    n_samples=n_samples,
                    num_runs=num_runs,
                    subtask_id=moa,
                    moa_split="same",
                    moa_name=moa,
                    use_drug_structure=use_drug_structure,
                )
            )
        if not specs:
            raise FileNotFoundError(f"No MOA train/test pairs under {root}")
        return specs


class MoaDiffTask(BaseTask):
    name = "moa_diff"

    def build_specs(self, **kwargs) -> List[TaskSpec]:
        kwargs.setdefault("output_dir", "runs/moa_diff")
        specs = MoaSameTask().build_specs(**kwargs)
        for spec in specs:
            spec.task_name = self.name
            spec.moa_split = "diff"
        return specs
