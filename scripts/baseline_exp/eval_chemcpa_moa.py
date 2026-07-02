"""Evaluate ChemCPA on PertDiffBench fig2 task1 MOA (SMILES + dose_value).

Predicts IFN response from control cells using test-set drug index + dose,
then reports the same 11 metrics as Squidiff / DDPM MOA scripts.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path


def _disable_broken_mpi4py() -> None:
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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from omegaconf import OmegaConf

from utils.metrics import (
    compute_des,
    compute_edistance,
    compute_mae,
    compute_mmd,
    compute_pds,
    compute_pearson,
    compute_pearson_delta,
    compute_pearson_delta_de,
    compute_r2,
)


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


def _dominant_drug_dose(test_path: str, drug_key: str, dose_key: str) -> tuple[str, float]:
    adata = sc.read_h5ad(test_path)
    ifn = adata[adata.obs["perturbation_status"] == "IFN"]
    if ifn.n_obs == 0:
        raise ValueError(f"No IFN cells in {test_path}")
    drugs = ifn.obs[drug_key].astype(str).tolist()
    doses = ifn.obs[dose_key].astype(float).tolist()
    dom_drug = Counter(drugs).most_common(1)[0][0]
    dom_dose = float(np.median(doses))
    dom_smiles = ""
    if "smiles" in ifn.obs.columns:
        sm_mask = ifn.obs[drug_key].astype(str) == dom_drug
        sm = ifn.obs.loc[sm_mask, "smiles"].astype(str).tolist()
        if sm:
            dom_smiles = Counter(sm).most_common(1)[0][0]
    return dom_drug, dom_dose, dom_smiles


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config",
        default=str(_project_root() / "configs/chemcpa/moa_fig2_task1.yaml"),
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-path", required=True, help="ChemCPA combined h5ad used for training")
    parser.add_argument("--test-data-path", required=True, help="Original test h5ad for metrics")
    parser.add_argument("--train-data-path", required=True, help="Original train h5ad for UMAP")
    parser.add_argument("-n", "--n_samples", type=int, default=0)
    parser.add_argument("-o", "--out_h5ad", required=True)
    parser.add_argument("--umap_plot", default="")
    parser.add_argument("--drug-key", default="perturbation")
    parser.add_argument("--dose-key", default="dose_value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (overridden by RUN_SEED env per run)")
    args = parser.parse_args()

    from utils.seed import resolve_seed, set_seed
    set_seed(resolve_seed(getattr(args, "seed", 0)))

    _setup_chemcpa_path()
    _disable_broken_mpi4py()
    from chemCPA.data.data import load_dataset_splits
    from chemCPA.lightning_module import ChemCPA

    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    cfg["dataset"]["data_params"]["dataset_path"] = args.data_path

    data_params = dict(cfg["dataset"]["data_params"])
    _, dataset = load_dataset_splits(**data_params, return_dataset=True)

    from utils.chemcpa_embeddings import save_drug_embeddings_parquet

    emb_path = Path(args.data_path).with_name(Path(args.data_path).stem + "_drug_emb.parquet")
    if not emb_path.exists():
        save_drug_embeddings_parquet(dataset.canon_smiles_unique_sorted, emb_path)
    cfg["model"]["embedding"]["datapath"] = str(emb_path)
    cfg["model"]["embedding"]["model"] = "rdkit"

    dataset_config = {
        "num_genes": dataset.num_genes,
        "num_drugs": dataset.num_drugs,
        "num_covariates": dataset.num_covariates,
        "use_drugs_idx": dataset.use_drugs_idx,
        "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
    }

    module = ChemCPA.load_from_checkpoint(
        args.ckpt,
        config=cfg,
        dataset_config=dataset_config,
        weights_only=False,
    )
    module.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    comp = module.model

    dom_drug, dom_dose, dom_smiles = _dominant_drug_dose(
        args.test_data_path, args.drug_key, args.dose_key
    )
    print(f"Target drug={dom_drug}, dose={dom_dose}, SMILES={dom_smiles}")

    if dom_drug not in dataset._drugs_name_to_idx:
        raise KeyError(f"Drug '{dom_drug}' not in training drug vocabulary: {dataset.drugs_names_unique_sorted}")
    drug_idx = dataset.drug_name_to_idx(dom_drug)

    adata_test = sc.read_h5ad(args.test_data_path)
    ctrl_mask = adata_test.obs["perturbation_status"] == "Control"
    pert_mask = adata_test.obs["perturbation_status"] == "IFN"
    ctrl_ids = adata_test.obs_names[ctrl_mask].tolist()
    pert_ids = adata_test.obs_names[pert_mask].tolist()
    if not pert_ids:
        print("No IFN cells. Exiting.")
        sys.exit(1)

    from utils.max_eval_samples import resolve_eval_n_samples
    n_samples = resolve_eval_n_samples(args.test_data_path, args.n_samples)
    selected_ctrl = np.random.choice(ctrl_ids, n_samples, replace=False)
    selected_pert = np.random.choice(pert_ids, n_samples, replace=False)

    ctrl_X = adata_test[selected_ctrl].X
    ctrl_X = ctrl_X.toarray() if hasattr(ctrl_X, "toarray") else ctrl_X
    ctrl_tensor = torch.from_numpy(ctrl_X.astype(np.float32)).to(device)

    max_pert = int(dataset.max_num_perturbations)
    drugs_idx = torch.full((n_samples, max_pert), 0, dtype=torch.long, device=device)
    drugs_idx[:, 0] = drug_idx
    dosages = torch.zeros((n_samples, max_pert), dtype=torch.float32, device=device)
    dosages[:, 0] = dom_dose

    cov_tensors = []
    if comp.num_covariates[0] > 0 and dataset.covariates is not None:
        ifn_obs = adata_test.obs[pert_mask]
        cov_col = cfg["dataset"]["data_params"]["covariate_keys"]
        if isinstance(cov_col, list):
            cov_col = cov_col[0]
        cov_names = dataset.covariate_names_unique[cov_col]
        cov_idx = 0
        if cov_col in ifn_obs.columns:
            dom_cov = Counter(ifn_obs[cov_col].astype(str).tolist()).most_common(1)[0][0]
            matches = np.where(cov_names == dom_cov)[0]
            if len(matches):
                cov_idx = int(matches[0])
        cov_oh = torch.zeros((n_samples, len(cov_names)), dtype=torch.float32, device=device)
        cov_oh[:, cov_idx] = 1.0
        cov_tensors = [cov_oh]

    with torch.no_grad():
        gene_recon, _ = comp.predict(
            genes=ctrl_tensor,
            drugs_idx=drugs_idx,
            dosages=dosages,
            covariates=cov_tensors,
            return_latent_basal=False,
        )
        dim = gene_recon.shape[1] // 2
        pred_pert = gene_recon[:, :dim].cpu().numpy()

    true_pert = adata_test[selected_pert].X
    true_pert = true_pert.toarray() if hasattr(true_pert, "toarray") else true_pert
    true_pert = true_pert.astype(np.float32)

    true_pb = np.mean(true_pert, axis=0)
    pred_pb = np.mean(pred_pert, axis=0)
    ctrl_pb = np.mean(ctrl_X, axis=0)

    mae = compute_mae(true_pb, pred_pb)
    r2 = compute_r2(true_pert, pred_pert)
    edist = compute_edistance(true_pert, pred_pert)
    mmd = compute_mmd(true_pert, pred_pert)
    p_all = compute_pearson(true_pb, pred_pb)
    pd_all = compute_pearson_delta(true_pb, pred_pb, ctrl_pb)
    pd20 = compute_pearson_delta_de(true_pb, pred_pb, ctrl_pb, k=20)
    pd50 = compute_pearson_delta_de(true_pb, pred_pb, ctrl_pb, k=50)
    pd100 = compute_pearson_delta_de(true_pb, pred_pb, ctrl_pb, k=100)

    delta_true = true_pb - ctrl_pb
    de_idx = np.argsort(np.abs(delta_true))[::-1][:100]
    true_de = set(adata_test.var_names[de_idx].tolist())
    delta_pred = pred_pb - ctrl_pb
    pred_de = set(adata_test.var_names[np.argsort(np.abs(delta_pred))[::-1][:100]].tolist())
    pred_fc = {g: fc for g, fc in zip(adata_test.var_names, delta_pred)}
    des = compute_des(true_de, pred_de, pred_fc)
    pds_val = compute_pds(np.array([pred_pb]), np.array([true_pb]))

    print("Perturbation Discrimination Score (PDS):", pds_val)
    print("Mean Absolute Error (MAE):", mae)
    print("Differential Expression Score (DES):", des)
    print("E-Distance:", edist)
    print("Maximum Mean Discrepancy (MMD):", mmd)
    print("R-squared (R2):", r2)
    print("Pearson (all genes):", p_all)
    print("Pearson Delta (all genes):", pd_all)
    print("Pearson Delta (top 20 DE genes):", pd20)
    print("Pearson Delta (top 50 DE genes):", pd50)
    print("Pearson Delta (top 100 DE genes):", pd100)

    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    obs = pd.DataFrame(
        {
            "perturbation_status": ["Predicted_IFN"] * n_samples,
            "origin_ctrl": selected_ctrl.tolist(),
            "perturbation": [dom_drug] * n_samples,
            "dose_value": [dom_dose] * n_samples,
            "smiles": [dom_smiles] * n_samples,
        },
        index=[f"synthetic_{i}" for i in range(n_samples)],
    )
    var = pd.DataFrame(index=adata_test.var_names)
    sc.AnnData(X=pred_pert, obs=obs, var=var).write_h5ad(args.out_h5ad)
    print("Saved", args.out_h5ad)

    if args.umap_plot:
        adata_train = sc.read_h5ad(args.train_data_path)
        adata_train.obs["data_source"] = "train"
        adata_test.obs["data_source"] = "test"
        adata_train.obs_names_make_unique()
        adata_test.obs_names_make_unique()
        adata_ref = sc.concat([adata_train, adata_test], join="outer", index_unique=None, fill_value=0)
        sc.pp.neighbors(adata_ref, n_neighbors=15, use_rep="X", random_state=0)
        sc.tl.umap(adata_ref, random_state=0)
        adata_synth = sc.AnnData(X=pred_pert, obs=obs, var=var)
        adata_synth.obs["data_source"] = "generated"
        adata_synth.obs_names_make_unique()
        adata_ref.raw = adata_ref
        adata_synth = adata_synth[:, adata_ref.var_names].copy()
        sc.tl.ingest(adata_synth, adata_ref, embedding_method="umap")
        adata_viz = sc.concat([adata_ref, adata_synth], join="inner", index_unique=None)
        adata_viz.obs["plot_group"] = "All Cells"
        adata_viz.obs.loc[
            (adata_viz.obs["perturbation_status"] != "Control") & (adata_viz.obs["data_source"] == "test"),
            "plot_group",
        ] = "True Perturbed"
        adata_viz.obs.loc[adata_viz.obs["data_source"] == "generated", "plot_group"] = "Generated Perturbed"
        adata_viz.obs["plot_group"] = pd.Categorical(
            adata_viz.obs["plot_group"],
            categories=["All Cells", "True Perturbed", "Generated Perturbed"],
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(
            adata_viz,
            color="plot_group",
            palette={
                "All Cells": "lightgray",
                "True Perturbed": "blue",
                "Generated Perturbed": "orange",
            },
            ax=ax,
            size=10,
            show=False,
        )
        os.makedirs(os.path.dirname(args.umap_plot) or ".", exist_ok=True)
        plt.savefig(args.umap_plot, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved UMAP to", args.umap_plot)


if __name__ == "__main__":
    main()
