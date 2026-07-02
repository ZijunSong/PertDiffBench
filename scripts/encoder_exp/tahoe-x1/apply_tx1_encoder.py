#!/usr/bin/env python3
"""
Apply a Tahoe-x1 (Tx1) encoder to an AnnData (.h5ad) file and write embeddings into adata.obsm.

Design goals (for your perturbation pipeline):
- Deterministic-ish inference: fixed seeds, no-grad
- Resume/skip: if output exists and already contains the embedding key, we skip by default
- Minimal assumptions about your AnnData: we try to manufacture required metadata if missing

Tx1 docs / APIs are still evolving, so this script attempts a few import paths:
  1) tahoe_x1.model.ComposerTX.from_hf(...) + (preferred) tahoe_x1.tasks.get_batch_embeddings
  2) scripts.inference.predict_embeddings.predict_embeddings (when running inside the tahoe-x1 repo)
  3) tahoe_x1.scripts.inference.predict_embeddings.predict_embeddings (if packaged)

If none are available, we fail loudly with guidance.

Example:
  python scripts/encoder_exp/apply_tx1_encoder.py \
      --data-path data/fig1/raw_task1/task1_valid_CD4T_exp.h5ad \
      --out-h5ad data/fig1/raw_task1/task1_valid_CD4T_exp_with_tx1_latent.h5ad \
      --model-size 70m \
      --hf-repo-id tahoebio/Tahoe-x1 \
      --obsm-key X_tx1 \
      --gpu
"""

from __future__ import annotations

import os
import argparse
import random
import tempfile
from typing import Optional, Tuple

import numpy as np
import scanpy as sc
import torch
from omegaconf import OmegaConf as om


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_required_fields(
    adata,
    cell_type_key: str,
    gene_id_key: str,
) -> None:
    """
    Tx1 embedding pipelines often expect:
      - adata.obs[cell_type_key]
      - adata.var[gene_id_key] (often Ensembl IDs)
    We try to fill them if missing to avoid hard crashes.
    """
    if cell_type_key not in adata.obs.columns:
        adata.obs[cell_type_key] = "unknown"
    if gene_id_key not in adata.var.columns:
        # Fallback: use var_names; this works ONLY if var_names match Tx1 vocab IDs.
        adata.var[gene_id_key] = adata.var_names.astype(str)


def _try_get_batch_embeddings_api():
    """
    Return (ComposerTX, get_batch_embeddings) if available, else (None, None).
    We use ComposerTX.from_hf() to load the model, so no separate load_model is needed.
    """
    try:
        from tahoe_x1.model import ComposerTX  # type: ignore
        from tahoe_x1.tasks import get_batch_embeddings  # type: ignore
        return ComposerTX, get_batch_embeddings
    except Exception:
        pass
    return None, None


def _encode_with_batch_embeddings(
    adata,
    hf_repo_id: str,
    model_size: str,
    obsm_key: str,
    return_gene_embeddings: bool,
    device: str,
    seq_len_dataset: int | None = None,
    batch_size: int | None = None,
    gene_id_key: str | None = None,
    **kwargs,
):
    """Extract embeddings using the packaged API (no need to run inside the Tahoe-x1 repo).

    We intentionally accept `seq_len_dataset` because many wrappers/plumbing pass it.
    Some Tahoe-x1 versions use this argument, others ignore it; we forward it only if supported.
    """
    import torch

    ComposerTX, get_batch_embeddings = _try_get_batch_embeddings_api()
    if ComposerTX is None or get_batch_embeddings is None:
        raise ImportError(
            "Tx1 embedding API not found. Expected ComposerTX in 'tahoe_x1.model' (or 'tahoex.model') and "
            "get_batch_embeddings in 'tahoex.tasks' (or 'tahoe_x1.tasks')."
        )

    model, vocab, model_cfg, collator_cfg = ComposerTX.from_hf(repo_id=hf_repo_id, model_size=model_size)  # type: ignore

    try:
        model.eval()
    except Exception:
        pass
    try:
        model.to(device)
    except Exception:
        pass

    # Prepare adata: Tx1 get_batch_embeddings expects adata.var["id_in_vocab"] and optionally subset to in-vocab genes (see predict_embeddings in tahoe-x1).
    adata_enc = adata
    gkey = gene_id_key or "ensembl_id"
    if gkey not in adata_enc.var.columns:
        gkey = adata_enc.var_names.name or "index"
        adata_enc.var[gkey] = adata_enc.var_names.astype(str)
    adata_enc.var["id_in_vocab"] = [
        vocab[g] if g in vocab else -1 for g in adata_enc.var[gkey]
    ]
    n_in_vocab = int(np.sum(np.array(adata_enc.var["id_in_vocab"]) >= 0))
    if n_in_vocab == 0:
        raise ValueError(
            f"No genes from adata.var['{gkey}'] found in Tx1 vocabulary. "
            "Ensure gene IDs match the model vocab (e.g. Ensembl IDs) or run add_ensembl_to_task1_h5ad.py."
        )
    adata_enc = adata_enc[:, np.array(adata_enc.var["id_in_vocab"]) >= 0].copy()
    gene_ids = np.array(
        [vocab[g] for g in adata_enc.var[gkey].tolist()],
        dtype=np.int64,
    )

    kwargs = dict(
        adata=adata_enc,
        model=model,
        vocab=vocab,
        model_cfg=model_cfg,
        collator_cfg=collator_cfg,
        gene_ids=gene_ids,
        return_gene_embeddings=return_gene_embeddings,
    )
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if seq_len_dataset is not None:
        kwargs["max_length"] = seq_len_dataset

    with torch.no_grad():
        out = get_batch_embeddings(**kwargs)

    if isinstance(out, tuple) and len(out) == 2:
        cell_embs, gene_embs = out
    else:
        cell_embs, gene_embs = out, None

    cell_embs = np.asarray(cell_embs, dtype=np.float32)
    if gene_embs is not None:
        gene_embs = np.asarray(gene_embs, dtype=np.float32)

    adata.obsm[obsm_key] = cell_embs
    return cell_embs, gene_embs

def _encode_with_predict_embeddings(
    adata,
    hf_repo_id: str,
    model_size: str,
    obsm_key: str,
    device: str,
    seq_len_dataset: int | None = None,
    gene_id_key: str | None = None,
    cell_type_key: str | None = None,
    return_gene_embeddings: bool = False,
    **kwargs,
):
    """Fallback path: use tahoe-x1 repo script `predict_embeddings.py` by loading it as a file.

    This is useful when:
      - you have the tahoe-x1 repo checked out (e.g., src/tahoe-x1/...), but
      - `scripts/` is not an importable python package (no __init__.py), so normal imports fail.
    """
    import os

    # Try to locate predict_embeddings.py
    cand = []

    # 1) User-provided env var pointing to tahoe-x1 repo root or to the script itself
    env_path = os.environ.get("TAHOEX1_PREDICT_PY", "")
    if env_path:
        cand.append(env_path)

    env_root = os.environ.get("TAHOEX1_ROOT", "")
    if env_root:
        cand.append(os.path.join(env_root, "scripts", "inference", "predict_embeddings.py"))

    # 2) Common relative path inside PertBench
    cand.append(os.path.join("src", "tahoe-x1", "scripts", "inference", "predict_embeddings.py"))

    # 3) Same dir guesses (in case this script was copied)
    cand.append(os.path.join(os.path.dirname(__file__), "predict_embeddings.py"))

    last_err = None
    for p in cand:
        try:
            predict_embeddings = _load_predict_embeddings_from_file(p)
            break
        except Exception as e:
            last_err = e
            predict_embeddings = None
    if predict_embeddings is None:
        raise ImportError(
            "predict_embeddings() not importable and file-based loading failed. "
            f"Tried: {cand}. Last error: {last_err}"
        )

    # Official API (tahoe-x1 README): predict_embeddings(cfg: DictConfig); cfg has paths, data, predict, model_name.
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        tmp_in = f.name
    try:
        adata.write_h5ad(tmp_in)
        cfg = om.create({
            "model_name": obsm_key,
            "paths": {
                "adata_input": os.path.abspath(tmp_in),
                "hf_repo_id": hf_repo_id,
                "hf_model_size": model_size,
            },
            "data": {
                "cell_type_key": cell_type_key or "cell_type",
                "gene_id_key": gene_id_key or "ensembl_id",
            },
            "predict": {
                "return_gene_embeddings": return_gene_embeddings,
                "seq_len_dataset": seq_len_dataset or 2048,
            },
        })
        om.resolve(cfg)
        result_adata = predict_embeddings(cfg)
        if obsm_key not in result_adata.obsm:
            raise RuntimeError(f"predict_embeddings returned adata but obsm['{obsm_key}'] not found")
        emb_arr = np.asarray(result_adata.obsm[obsm_key], dtype=np.float32)
        # Align to original adata rows (predict_embeddings may drop rows with na cell_type)
        if result_adata.n_obs == adata.n_obs and (result_adata.obs.index == adata.obs.index).all():
            adata.obsm[obsm_key] = emb_arr
        else:
            idx = result_adata.obs.index.get_indexer(adata.obs.index)
            adata.obsm[obsm_key] = emb_arr[idx]
        return np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    finally:
        if os.path.isfile(tmp_in):
            os.remove(tmp_in)

def _load_predict_embeddings_from_file(py_path: str):
    """Load `predict_embeddings` function from a python file path (repo script) via importlib.

    This avoids relying on `scripts.*` being an importable Python package.
    """
    import importlib.util
    import os

    if not py_path:
        raise FileNotFoundError("Empty predict_embeddings path.")
    py_path = os.path.abspath(py_path)
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"predict_embeddings.py not found: {py_path}")

    spec = importlib.util.spec_from_file_location("tx1_predict_embeddings_mod", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "predict_embeddings"):
        raise ImportError(f"No predict_embeddings() in: {py_path}")
    return getattr(mod, "predict_embeddings")


def main():
    p = argparse.ArgumentParser(description="Apply Tahoe-x1 encoder and write cell embeddings into AnnData.obsm.")
    p.add_argument("--data-path", type=str, required=True, help="Input AnnData (.h5ad).")
    p.add_argument("--out-h5ad", type=str, required=True, help="Output AnnData (.h5ad) with embeddings in .obsm.")
    p.add_argument("--hf-repo-id", type=str, default="tahoebio/Tahoe-x1", help="Hugging Face repo id for Tx1.")
    p.add_argument("--model-size", type=str, default="70m", choices=["70m", "1b", "3b"], help="Tx1 model size.")
    p.add_argument("--obsm-key", type=str, default="X_tx1", help="Key to store embeddings in adata.obsm.")
    p.add_argument("--cell-type-key", type=str, default="cell_type", help="adata.obs column for cell type.")
    p.add_argument("--gene-id-key", type=str, default="ensembl_id", help="adata.var column for gene ids.")
    p.add_argument("--seq-len-dataset", type=int, default=2048, help="Max sequence length for tokenizer/dataset.")
    p.add_argument("--batch-size", type=int, default=64, help="Embedding extraction batch size (if supported).")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    p.add_argument("--force", action="store_true", help="Overwrite even if output already exists with embeddings.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = p.parse_args()

    from utils.seed import resolve_seed
    seed_everything(resolve_seed(args.seed))

    if os.path.exists(args.out_h5ad) and (not args.force):
        try:
            out = sc.read_h5ad(args.out_h5ad)
            if args.obsm_key in out.obsm:
                print(f"[tx1-encode] Found existing output with obsm['{args.obsm_key}']; skip: {args.out_h5ad}")
                return
        except Exception:
            # If output exists but unreadable, we will re-run.
            pass

    print(f"[tx1-encode] Loading AnnData: {os.path.abspath(args.data_path)}")
    adata = sc.read_h5ad(args.data_path)

    _ensure_required_fields(adata, args.cell_type_key, args.gene_id_key)

    use_gpu = bool(args.gpu and torch.cuda.is_available())
    device = "cuda" if use_gpu else "cpu"
    print(f"[tx1-encode] device={device}  model={args.hf_repo_id}  size={args.model_size}")

    # Try "batch embeddings" API first; fall back to predict_embeddings.
    embs = None
    last_err = None
    try:
        embs, _ = _encode_with_batch_embeddings(
            adata=adata,
            device=device,
            hf_repo_id=args.hf_repo_id,
            model_size=args.model_size,
            obsm_key=args.obsm_key,
            seq_len_dataset=args.seq_len_dataset,
            return_gene_embeddings=False,
            batch_size=args.batch_size,
            gene_id_key=args.gene_id_key,
        )
        print("[tx1-encode] Encoded via get_batch_embeddings API.")
    except Exception as e:
        last_err = e
        print(f"[tx1-encode] get_batch_embeddings path failed: {type(e).__name__}: {e}")
        try:
            embs = _encode_with_predict_embeddings(
                adata=adata,
                hf_repo_id=args.hf_repo_id,
                model_size=args.model_size,
                obsm_key=args.obsm_key,
                device=device,
                seq_len_dataset=args.seq_len_dataset,
                cell_type_key=args.cell_type_key,
                gene_id_key=args.gene_id_key,
                return_gene_embeddings=False,
            )
            print("[tx1-encode] Encoded via predict_embeddings fallback.")
        except Exception as e2:
            raise RuntimeError(
                "Failed to extract Tx1 embeddings via all known APIs.\n"
                f"1) get_batch_embeddings error: {last_err}\n"
                f"2) predict_embeddings error: {e2}\n"
                "Fix hints:\n"
                "  - Ensure Tahoe-x1 is installed (recommended: clone repo and `pip install -e .`).\n"
                "  - Verify your AnnData has gene ids matching Tx1 vocab (often Ensembl IDs).\n"
                "  - See README section 'Generating Cell and Gene Embeddings' in tahoebio/tahoe-x1.\n"
            ) from e2

    print(f"[tx1-encode] Embedding shape = {np.asarray(embs).shape}  stored at obsm['{args.obsm_key}'].")

    os.makedirs(os.path.dirname(args.out_h5ad) or ".", exist_ok=True)
    adata.write_h5ad(args.out_h5ad)
    print(f"[tx1-encode] ✔ Saved: {args.out_h5ad}")


if __name__ == "__main__":
    main()
