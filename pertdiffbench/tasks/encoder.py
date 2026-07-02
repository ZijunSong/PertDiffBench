from __future__ import annotations

from pathlib import Path
from typing import List

from pertdiffbench.tasks.base import BaseTask, TaskSpec

ENCODER_NAMES = (
    "scgpt",
    "scfoundation",
    "scvi",
    "geneformer",
    "state",
    "scimilarity",
    "cellfm",
    "tahoe_x1",
)

LATENT_KEYS = {
    "scgpt": "X_scgpt",
    "scfoundation": "X_scfoundation",
    "scvi": "X_scvi",
    "geneformer": "X_geneformer",
    "state": "X_state",
    "scimilarity": "X_scimilarity",
    "cellfm": "X_cellfm",
    "tahoe_x1": "X_tahoe_x1",
}


class EncoderTask(BaseTask):
    """Pretrained encoder + latent DDPM on known-condition CD4T-style splits."""

    name = "encoder"

    def build_specs(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        output_dir: str | Path = "runs/encoder",
        encoder_name: str = "scgpt",
        gene_nums: int = 6998,
        n_samples: int = 0,
        num_runs: int = 3,
        encoder_ckpt_dir: str | Path | None = None,
        **_,
    ) -> List[TaskSpec]:
        if encoder_name not in ENCODER_NAMES:
            raise ValueError(f"encoder_name must be one of {ENCODER_NAMES}")
        return [
            TaskSpec(
                task_name=self.name,
                train_h5ad=train_h5ad,
                test_h5ad=test_h5ad,
                output_dir=Path(output_dir) / encoder_name,
                gene_nums=gene_nums,
                n_samples=n_samples,
                num_runs=num_runs,
                subtask_id=encoder_name,
                encoder_name=encoder_name,
                latent_key=LATENT_KEYS[encoder_name],
                encoder_ckpt_dir=Path(encoder_ckpt_dir) if encoder_ckpt_dir else None,
            )
        ]
