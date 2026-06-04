"""Train ChemCPA on PertDiffBench fig2 task1 MOA (SMILES + dose_value).

Requires chemCPA installed from https://github.com/theislab/chemCPA
  git clone https://github.com/theislab/chemCPA.git src/chemCPA
  cd src/chemCPA && pip install -e .

Set CHEMCPA_ROOT to the chemCPA repo root if not at src/chemCPA.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _disable_broken_mpi4py() -> None:
    """Avoid Lightning MPI auto-detect when mpi4py is installed without libmpi."""
    try:
        from mpi4py import MPI  # noqa: F401
    except (ImportError, RuntimeError):
        from unittest.mock import MagicMock

        mock_mpi = MagicMock()
        mock_comm = MagicMock()
        mock_comm.Get_size.return_value = 1
        mock_mpi.COMM_WORLD = mock_comm
        sys.modules["mpi4py"] = MagicMock(MPI=mock_mpi)
        sys.modules["mpi4py.MPI"] = mock_mpi


_disable_broken_mpi4py()

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_chemcpa_path() -> Path:
    root = Path(os.environ.get("CHEMCPA_ROOT", _project_root() / "src" / "chemCPA")).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"ChemCPA not found at {root}. Clone https://github.com/theislab/chemCPA "
            f"to src/chemCPA or set CHEMCPA_ROOT."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config",
        default=str(_project_root() / "configs/chemcpa/moa_fig2_task1.yaml"),
    )
    parser.add_argument("--data-path", required=True, help="ChemCPA-format combined h5ad")
    parser.add_argument("--save-dir", required=True, help="Checkpoint output directory")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    _setup_chemcpa_path()
    from chemCPA.data.data import PerturbationDataModule, load_dataset_splits
    from chemCPA.lightning_module import ChemCPA

    L.seed_everything(args.seed, workers=True)
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)

    cfg["dataset"]["data_params"]["dataset_path"] = args.data_path
    cfg["training"]["save_dir"] = args.save_dir
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        cfg["model"]["hparams"]["batch_size"] = args.batch_size

    data_params = dict(cfg["dataset"]["data_params"])
    datasets, dataset = load_dataset_splits(**data_params, return_dataset=True)

    from utils.chemcpa_embeddings import save_drug_embeddings_parquet

    emb_path = Path(args.data_path).with_name(Path(args.data_path).stem + "_drug_emb.parquet")
    if not emb_path.exists():
        save_drug_embeddings_parquet(dataset.canon_smiles_unique_sorted, emb_path)
    cfg["model"]["embedding"]["datapath"] = str(emb_path)
    cfg["model"]["embedding"]["model"] = "rdkit"

    dm_kwargs = {
        "datasplits": datasets,
        "train_bs": int(cfg["model"]["hparams"]["batch_size"]),
    }
    try:
        dm = PerturbationDataModule(num_workers=args.num_workers, **dm_kwargs)
    except TypeError:
        dm = PerturbationDataModule(**dm_kwargs)

    dataset_config = {
        "num_genes": datasets["training"].num_genes,
        "num_drugs": datasets["training"].num_drugs,
        "num_covariates": datasets["training"].num_covariates,
        "use_drugs_idx": dataset.use_drugs_idx,
        "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
    }
    dataset.debug_print()

    model = ChemCPA(cfg, dataset_config)

    ckpt_path = Path(args.save_dir) / "last.ckpt"
    if ckpt_path.exists():
        print(f"Checkpoint exists, skipping training: {ckpt_path}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="last",
        save_last=True,
        save_top_k=1,
    )
    callbacks = [checkpoint_callback]

    max_minutes = int(cfg["training"].get("max_minutes") or 0)
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        max_epochs=int(cfg["training"]["num_epochs"]),
        max_time={"minutes": max_minutes} if max_minutes > 0 else None,
        callbacks=callbacks,
        check_val_every_n_epoch=int(cfg["training"]["checkpoint_freq"]),
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)
    print(f"Training complete. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
