"""Command-line interface for PertDiffBench."""

from __future__ import annotations

import argparse
import sys

from pertdiffbench import __version__
from pertdiffbench.registry import SUPPORTED_MODELS
from pertdiffbench.runner import BenchmarkRunner
from pertdiffbench.tasks.registry import SUPPORTED_TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pertdiffbench",
        description="Unified wrapper for perturbation-response model benchmarking.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Train and evaluate one or more models")
    run.add_argument(
        "--task",
        default="known_condition",
        choices=SUPPORTED_TASKS,
        help="Benchmark task",
    )
    run.add_argument("--train", help="Path to training .h5ad")
    run.add_argument("--test", help="Path to test .h5ad")
    run.add_argument("--combined", help="Combined h5ad for scGen-style tasks")
    run.add_argument("--data-root", help="Data root (MOA splits, noise, etc.)")
    run.add_argument(
        "--models",
        default=",".join(m for m in SUPPORTED_MODELS if m != "encoder"),
        help="Comma-separated model names",
    )
    run.add_argument("--output", default="runs/exp", help="Output directory")
    run.add_argument("--gene-nums", type=int, default=1000)
    run.add_argument("--n-samples", type=int, default=100)
    run.add_argument("--num-runs", type=int, default=3)
    run.add_argument("--repo-root", default=None)
    run.add_argument("--skip-train", action="store_true")

    run.add_argument("--celltype", help="Cell type to predict (cross_celltype tasks)")
    run.add_argument("--species", help="Held-out species (cross_species_loo)")
    run.add_argument("--held-out-celltype", help="LOO cell type (cross_celltype_plus)")
    run.add_argument("--control-fraction", default="p0.25", help="p0 | p0.25 | p0.5")
    run.add_argument("--moa-name", help="Single MOA name (optional filter)")
    run.add_argument("--noise-type", choices=["gaussian", "lognormal", "poisson", "zero_inflation"])
    run.add_argument("--noise-level", help="Noise level string, e.g. 0.5")
    run.add_argument(
        "--encoder",
        choices=["scgpt", "scfoundation", "scvi", "geneformer", "state", "scimilarity", "cellfm", "tahoe_x1"],
        help="Encoder name (required for --task encoder)",
    )
    run.add_argument("--encoder-ckpt-dir", help="Pretrained encoder checkpoint directory")
    run.add_argument("--annotation-model-dir", help="scDiffusion VAE annotation model dir")
    run.add_argument("--scdiff-data-root", help="Directory containing scDiff h5ad files")
    run.add_argument("--scdiff-dataset", help="scDiff dataset name override")
    run.add_argument("--use-drug-structure", action="store_true", help="MOA drug SMILES conditioning")

    sub.add_parser("list-models", help="List supported models")
    list_models = sub.add_parser("list-models-for-task", help="List models for a task")
    list_models.add_argument("--task", default="known_condition", choices=SUPPORTED_TASKS)
    sub.add_parser("list-tasks", help="List supported tasks")
    return parser


def _build_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "output_dir": args.output,
        "gene_nums": args.gene_nums,
        "n_samples": args.n_samples,
        "num_runs": args.num_runs,
    }
    if args.train:
        kwargs["train_h5ad"] = args.train
    if args.test:
        kwargs["test_h5ad"] = args.test
    if args.combined:
        kwargs["combined_h5ad"] = args.combined
    if args.data_root:
        kwargs["data_root"] = args.data_root
    if args.celltype:
        kwargs["celltype_to_predict"] = args.celltype
    if args.species:
        kwargs["held_out_species"] = args.species
        kwargs["species_to_predict"] = args.species
    if args.held_out_celltype:
        kwargs["held_out_celltype"] = args.held_out_celltype
    if args.control_fraction:
        kwargs["control_fraction"] = args.control_fraction
    if args.moa_name:
        kwargs["moa_name"] = args.moa_name
    if args.noise_type:
        kwargs["noise_type"] = args.noise_type
    if args.noise_level:
        kwargs["noise_level"] = args.noise_level
    if args.encoder:
        kwargs["encoder_name"] = args.encoder
    if args.encoder_ckpt_dir:
        kwargs["encoder_ckpt_dir"] = args.encoder_ckpt_dir
    if args.use_drug_structure:
        kwargs["use_drug_structure"] = True
    return kwargs


def _validate_run_args(args: argparse.Namespace) -> None:
    task = args.task
    if task in {"moa_same", "moa_diff"}:
        if not args.data_root and not (args.train and args.test):
            raise SystemExit("--data-root or (--train and --test) required for MOA tasks")
    elif task == "encoder":
        if not args.encoder:
            raise SystemExit("--encoder is required for --task encoder")
        if not args.train or not args.test:
            raise SystemExit("--train and --test required for encoder task")
    elif task == "cross_celltype_extend":
        if not args.combined or not args.test:
            raise SystemExit("--combined and --test required for cross_celltype_extend")
    elif task == "cross_celltype_plus":
        if not all([args.train, args.test, args.combined, args.held_out_celltype]):
            raise SystemExit("--train, --test, --combined, --held-out-celltype required")
    elif task == "noise":
        if not args.train or not args.test or not args.noise_type:
            raise SystemExit("--train, --test, --noise-type required for noise task")
    else:
        if not args.train or not args.test:
            raise SystemExit("--train and --test are required for this task")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-tasks":
        for t in SUPPORTED_TASKS:
            print(f"  - {t}")
        return 0

    if args.command == "list-models":
        print("Supported models:")
        for name in SUPPORTED_MODELS:
            print(f"  - {name}")
        return 0

    if args.command == "list-models-for-task":
        runner = BenchmarkRunner(task=args.task)
        print(f"Models for task '{args.task}':")
        for name in runner.list_models(args.task):
            print(f"  - {name}")
        return 0

    if args.command == "run":
        _validate_run_args(args)
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        runner = BenchmarkRunner(
            task=args.task,
            output_dir=args.output,
            repo_root=args.repo_root,
        )
        build_kwargs = _build_kwargs(args)
        if args.annotation_model_dir:
            from pathlib import Path
            build_kwargs["annotation_model_dir"] = Path(args.annotation_model_dir)
        if args.scdiff_data_root:
            from pathlib import Path
            build_kwargs["scdiff_data_root"] = Path(args.scdiff_data_root)
        if args.scdiff_dataset:
            build_kwargs["scdiff_dataset"] = args.scdiff_dataset

        runner.run(models=models, skip_train=args.skip_train, **build_kwargs)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
