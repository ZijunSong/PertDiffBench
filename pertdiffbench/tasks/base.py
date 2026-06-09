from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TaskSpec:
    """Single train/eval configuration for one backend run."""

    task_name: str
    train_h5ad: Path
    test_h5ad: Path
    output_dir: Path
    gene_nums: int = 1000
    n_samples: int = 100
    num_runs: int = 3
    subtask_id: str = "default"

    ckpt_dir: Optional[Path] = None
    sample_dir: Optional[Path] = None

    combined_h5ad: Optional[Path] = None
    pair_only_obs_key: Optional[str] = None
    pair_only_obs_value: Optional[str] = None

    moa_split: Optional[str] = None
    moa_name: Optional[str] = None
    use_drug_structure: bool = False
    drug_key: str = "perturbation"
    dose_key: str = "dose_value"

    celltype_to_predict: Optional[str] = None
    species_to_predict: Optional[str] = None

    scdiff_data_root: Optional[Path] = None
    scdiff_dataset: Optional[str] = None
    scdiff_train_fname: Optional[str] = None
    scdiff_test_fname: Optional[str] = None

    time_key: str = "treatment_time"

    noise_type: Optional[str] = None
    noise_level: Optional[str] = None

    encoder_name: Optional[str] = None
    encoder_ckpt_dir: Optional[Path] = None
    latent_key: Optional[str] = None

    annotation_model_dir: Optional[Path] = None
    chemcpa_combined_h5ad: Optional[Path] = None

    train_once: bool = False
    squidiff_train_once: bool = False
    skip_umap: bool = True

    def __post_init__(self) -> None:
        self.train_h5ad = Path(self.train_h5ad)
        self.test_h5ad = Path(self.test_h5ad)
        self.output_dir = Path(self.output_dir)
        if self.ckpt_dir is None:
            self.ckpt_dir = self.output_dir / "checkpoints" / self.subtask_id
        else:
            self.ckpt_dir = Path(self.ckpt_dir)
        if self.sample_dir is None:
            self.sample_dir = self.output_dir / "samples" / self.subtask_id
        else:
            self.sample_dir = Path(self.sample_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        if self.combined_h5ad is not None:
            self.combined_h5ad = Path(self.combined_h5ad)
        if self.scdiff_data_root is not None:
            self.scdiff_data_root = Path(self.scdiff_data_root)
        if self.annotation_model_dir is not None:
            self.annotation_model_dir = Path(self.annotation_model_dir)
        if self.chemcpa_combined_h5ad is not None:
            self.chemcpa_combined_h5ad = Path(self.chemcpa_combined_h5ad)
        if self.encoder_ckpt_dir is not None:
            self.encoder_ckpt_dir = Path(self.encoder_ckpt_dir)


class BaseTask(ABC):
    name: str = "base"

    @abstractmethod
    def build_specs(self, **kwargs) -> List[TaskSpec]:
        raise NotImplementedError

    def validate(self, spec: TaskSpec) -> None:
        if not spec.train_h5ad.exists():
            raise FileNotFoundError(f"Training data not found: {spec.train_h5ad}")
        if not spec.test_h5ad.exists():
            raise FileNotFoundError(f"Test data not found: {spec.test_h5ad}")
