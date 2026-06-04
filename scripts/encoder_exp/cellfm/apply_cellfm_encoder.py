# scripts/encoder_exp/cellfm/apply_cellfm_encoder.py
#!/usr/bin/env python3
"""
Apply a (pretrained) CellFM encoder to an AnnData file and write X_cellfm.

Uses the MindSpore CellFM from src/CellFM (80M checkpoint). Expects checkpoint
at --ckpt-path (e.g. CellFM_80M_weight.ckpt). Gene space is aligned via
src/CellFM/csv/gene_info.csv.

Usage example:
    python scripts/encoder_exp/cellfm/apply_cellfm_encoder.py \
        --data-path data/fig1/raw_task1/task1_train_CD4T_exp.h5ad \
        --out-h5ad samples/encoder_exp/cellfm_ddpm/task1_train_CD4T_with_cellfm_latent.h5ad \
        --ckpt-path /path/to/CellFM_80M_weight.ckpt \
        --device cuda
"""

import os
import sys
import argparse
import traceback
from typing import Tuple

# in import other before MindSpore/GLOG ( load_param_into_net "parameters not loaded")
if "GLOG_minloglevel" not in os.environ:
    os.environ["GLOG_minloglevel"] = "2"  # 0=INFO, 1=WARNING, 2=ERROR


def _ensure_ld_path_for_gpu_and_reexec():
    """
    Linux under in when LD_LIBRARY_PATH, in Python after .
     mustusing GPU, in Python before ; herevia re-exec correct LD_LIBRARY_PATH.
    """
    # checkwhethermust GPU
    need_gpu = False
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].startswith("cuda"):
            need_gpu = True
    else:
        # default device cuda, asmust GPU
        need_gpu = True
    if not need_gpu:
        return
    # conda path (and _setup_mindspore_gpu_env )
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix and hasattr(sys, "executable"):
        exe = os.path.realpath(sys.executable)
        if os.path.sep + "envs" + os.path.sep in exe:
            parts = exe.split(os.path.sep)
            try:
                i = parts.index("envs")
                if i + 1 < len(parts):
                    conda_prefix = os.path.sep.join(parts[: i + 2])
            except ValueError:
                pass
    if not conda_prefix or not os.path.isdir(os.path.join(conda_prefix, "lib")):
        # : to conda_prefix, batch info
        if not conda_prefix:
            print(f"[apply_cellfm] DEBUG: CONDA_PREFIX not found, sys.executable={sys.executable if hasattr(sys, 'executable') else 'N/A'}", file=sys.stderr, flush=True)
        return
    conda_lib = os.path.join(conda_prefix, "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    # checkwhether correct (conda lib in front)
    # : LD_LIBRARY_PATH correct, MindSpore run_check in import when , 
    # when using when LD_LIBRARY_PATH.
    # LD_LIBRARY_PATH in when (e.g.in Python ), 
    # wemust re-exec toEnsure in when correct LD_LIBRARY_PATH.
    # as , we in mustwhen re-exec ( LD_LIBRARY_PATH correct when).
    
    # checkexistingwhethercontainMindSporepath ( alreadyimport MindSpore)
    has_mindspore_path = "mindspore/lib/plugin" in existing
    
    # correctLD_LIBRARY_PATH: conda libin front, after lib
    new_ld_parts = [conda_lib, "/usr/lib/x86_64-linux-gnu"]
    # fromexisting innew_ld_parts path, CUDApath MindSporepath
    if existing:
        existing_parts = existing.split(os.pathsep)
        for p in existing_parts:
            if p and p not in new_ld_parts and not p.startswith("/usr/local/cuda") and "mindspore" not in p:
                new_ld_parts.append(p)
    new_ld = os.pathsep.join(new_ld_parts)
    
    # checkwhethermustre-exec: 
    # 1. LD_LIBRARY_PATHnot set
    # 2. conda lib in front
    # 3. containMindSporepath ( alreadyimport , mustre-exec )
    needs_reexec = (not existing or 
                    (not existing.startswith(conda_lib + os.pathsep) and existing != conda_lib) or
                    has_mindspore_path)
    
    if needs_reexec:
        # must re-exec: LD_LIBRARY_PATH andkeep CONDA_PREFIX
        os.environ["LD_LIBRARY_PATH"] = new_ld
        if not os.environ.get("CONDA_PREFIX"):
            os.environ["CONDA_PREFIX"] = conda_prefix
        # Re-exec: willfrom , LD_LIBRARY_PATH in whencorrect
        # : execv will current , to print will inoutput 
        # to , re-exec (execv )
        print(f"[apply_cellfm] Re-executing Python with LD_LIBRARY_PATH={new_ld[:150]}...", file=sys.stderr, flush=True)
        print(f"[apply_cellfm] DEBUG: existing={existing[:100]}, has_mindspore={has_mindspore_path}, needs_reexec={needs_reexec}", file=sys.stderr, flush=True)
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[apply_cellfm] Re-exec failed: {e}", file=sys.stderr, flush=True)
            raise
    else:
        # correct , EnsureLD_LIBRARY_PATHcontainall mustpath
        # containMindSporepath, stillmustre-exec 
        if has_mindspore_path:
            print(f"[apply_cellfm] DEBUG: LD_LIBRARY_PATH contains MindSpore path, forcing re-exec", file=sys.stderr, flush=True)
            os.environ["LD_LIBRARY_PATH"] = new_ld
            if not os.environ.get("CONDA_PREFIX"):
                os.environ["CONDA_PREFIX"] = conda_prefix
            print(f"[apply_cellfm] Re-executing Python with LD_LIBRARY_PATH={new_ld[:150]}...", file=sys.stderr, flush=True)
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                print(f"[apply_cellfm] Re-exec failed: {e}", file=sys.stderr, flush=True)
                raise
        else:
            os.environ["LD_LIBRARY_PATH"] = new_ld


# need GPU, Ensure LD_LIBRARY_PATH in beforecontain conda/lib, mustwhen re-exec
_ensure_ld_path_for_gpu_and_reexec()

import numpy as np
import pandas as pd
import scanpy as sc

# Device is used for MindSpore below; torch only for CUDA check if needed
try:
    import torch
except Exception:
    torch = None


# Nonz length and pad from CellFM config (80M)
NONZ_LEN = 2048
PAD = 1
CELLFM_LATENT_DIM = 1536


def _setup_cellfm_path(repo_root: str) -> None:
    cellfm_src = os.path.join(repo_root, "src", "CellFM")
    if os.path.isdir(cellfm_src) and cellfm_src not in sys.path:
        sys.path.insert(0, cellfm_src)


def _get_repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts/encoder_exp/cellfm -> ../../../ = repo root
    return os.path.abspath(os.path.join(script_dir, "..", "..", ".."))


def _resolve_ckpt_path(ckpt_path: str) -> str:
    """
    parse checkpoint path: as symlink ( HF blobs ), 
     from HuggingFace under to directoryunder fileandreturncanusingpath.
    """
    ckpt_path = os.path.abspath(ckpt_path)
    # currentpathcan (or symlink exist), directly using
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        return ckpt_path
    real = os.path.realpath(ckpt_path)
    if os.path.isfile(real) and os.path.getsize(real) > 0:
        return real
    # symlink or exist: from HuggingFace under to directory file
    ckpt_dir = os.path.dirname(ckpt_path)
    local_ckpt = os.path.join(ckpt_dir, "CellFM_80M_weight_local.ckpt")
    if os.path.isfile(local_ckpt) and os.path.getsize(local_ckpt) > 0:
        return local_ckpt
    try:
        from huggingface_hub import hf_hub_download
        print("[apply_cellfm] Checkpoint path missing or broken symlink, downloading from HuggingFace...")
        # under to file, to blobs symlink
        downloaded = hf_hub_download(
            repo_id="ShangguanNingyuan/CellFM",
            filename="CellFM_80M_weight.ckpt",
            local_dir=ckpt_dir,
            local_dir_use_symlinks=False,
        )
        return os.path.abspath(downloaded)
    except Exception as e:
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path} (resolved: {real}) and download failed: {e}. "
            "Please ensure the file exists or install huggingface_hub and retry."
        ) from e


def _load_geneset(repo_root: str, use_expand: bool = True):
    """
    Load gene name -> 1-based index. CellFM_80M using expand_gene_info train (27855 gene), 
    needand checkpoint gene_emb shape (27856, 1536) , defaultusing expand_gene_info.csv.
    """
    filename = "expand_gene_info.csv" if use_expand else "gene_info.csv"
    path = os.path.join(repo_root, "src", "CellFM", "csv", filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CellFM {filename} not found at {path}. "
            f"Ensure repo contains src/CellFM/csv/{filename}."
        )
    df = pd.read_csv(path, index_col=0, header=0)
    geneset = {name: (i + 1) for i, name in enumerate(df.index)}
    return geneset


def _prepare_cell_batch(
    X_block: np.ndarray,
    common_genes: list,
    geneset: dict,
    total_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (expr, gene, zero_idx) for a batch of cells.
    X_block: (n_cells, n_genes), dense; total_counts: (n_cells,).
    Returns expr (n_cells, NONZ_LEN, 2) - ValueEncoder must [unmask, expr] , when unmask=1; 
    gene (n_cells, NONZ_LEN), zero_idx (n_cells, NONZ_LEN + PAD).
    """
    n_cells = X_block.shape[0]
    expr_list = []
    gene_list = []
    zero_idx_list = []

    for i in range(n_cells):
        row = np.asarray(X_block[i], dtype=np.float32).ravel()
        read = float(max(1, total_counts[i] / 1e5))
        nonz = np.where(row > 0)[0]
        if len(nonz) == 0:
            expr_vec = np.zeros((NONZ_LEN, 2), dtype=np.float32)
            gene_vec = np.zeros(NONZ_LEN, dtype=np.int32)
            zero_idx = np.zeros(NONZ_LEN + PAD, dtype=np.float32)
            zero_idx[0] = 1.0  # cls only
        else:
            vals = row[nonz]
            norm_vals = np.log1p(vals / read * 1e4).astype(np.float32)
            order = np.argsort(-norm_vals)
            nonz = nonz[order]
            norm_vals = norm_vals[order]
            seq_len = min(len(nonz), NONZ_LEN)
            gene_ids = np.array(
                [geneset[common_genes[j]] for j in nonz[:seq_len]],
                dtype=np.int32,
            )
            expr_vec = np.zeros((NONZ_LEN, 2), dtype=np.float32)
            expr_vec[:seq_len, 0] = 1.0
            expr_vec[:seq_len, 1] = norm_vals[:seq_len]
            gene_vec = np.zeros(NONZ_LEN, dtype=np.int32)
            gene_vec[:seq_len] = gene_ids
            zero_idx = np.zeros(NONZ_LEN + PAD, dtype=np.float32)
            zero_idx[: seq_len + 1] = 1.0

        expr_list.append(expr_vec)
        gene_list.append(gene_vec)
        zero_idx_list.append(zero_idx)

    expr = np.stack(expr_list, axis=0)
    gene = np.stack(gene_list, axis=0)
    zero_idx = np.stack(zero_idx_list, axis=0)
    return expr, gene, zero_idx


def _setup_mindspore_gpu_env():
    """in import mindspore before LD_LIBRARY_PATH, mindspore-gpu found libcuda/libcudnn (with conda )."""
    extra_paths = []
    # using conda CUDA (using sudo whenusing conda install cuda-toolkit cudnn)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix and hasattr(sys, "executable"):
        # nohup under CONDA_PREFIX may , from python path ( .../envs/cellfm/bin/python)
        exe = os.path.realpath(sys.executable)
        if os.path.sep + "envs" + os.path.sep in exe:
            # .../envs/ENVNAME/bin/python -> .../envs/ENVNAME
            parts = exe.split(os.path.sep)
            try:
                idx = parts.index("envs")
                if idx + 1 < len(parts):
                    conda_prefix = os.path.sep.join(parts[: idx + 2])
            except ValueError:
                pass
    if conda_prefix and os.path.isdir(os.path.join(conda_prefix, "lib")):
        extra_paths.append(os.path.join(conda_prefix, "lib"))
    # libcuda in /usr/lib/x86_64-linux-gnu
    extra_paths.append("/usr/lib/x86_64-linux-gnu")
    # only conda CUDA when CUDA, to libcublas.so.12/13
    if not (conda_prefix and os.path.isdir(os.path.join(conda_prefix, "lib"))):
        extra_paths.extend([
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
        ])
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    added = os.pathsep.join(p for p in extra_paths if os.path.isdir(p))
    if added:
        # Ensureconda libin front, to CUDA 13
        existing_parts = [p for p in existing.split(os.pathsep) if p and p not in extra_paths]
        # mayexist CUDApath
        existing_parts = [p for p in existing_parts if not p.startswith("/usr/local/cuda")]
        new_ld = added
        if existing_parts:
            new_ld = new_ld + os.pathsep + os.pathsep.join(existing_parts)
        os.environ["LD_LIBRARY_PATH"] = new_ld


def encode_with_cellfm(adata, ckpt_path: str, device: str = "cpu") -> np.ndarray:
    """
    Encode cells with CellFM (MindSpore 80M) and return cell embeddings (n_cells, 1536).
     using GPU before mindspore-gpu ( or README), andEnsure libcuda.so, libcudnn.so.
    """
    ckpt_path = _resolve_ckpt_path(ckpt_path)
    repo_root = _get_repo_root()
    _setup_cellfm_path(repo_root)

    # in import before , to mindspore-gpu when found CUDA/cuDNN
    _setup_mindspore_gpu_env()

    import mindspore as ms
    import logging as _logging
    # MindSpore : load_param_into_net "456 parameters not loaded" 
    for _name in ("mindspore", "mindspore.train", "mindspore.train.serialization", "mindspore.context", "mindspore.run_check"):
        _logging.getLogger(_name).setLevel(_logging.ERROR)

    if device.startswith("cuda"):
        try:
            ms.set_context(device_target="GPU", device_id=0)
        except Exception as e:
            print(
                "[apply_cellfm] GPU context failed (MindSpore could not load GPU).",
                file=sys.stderr,
            )
            print(f"[apply_cellfm] MindSpore : {e}", file=sys.stderr)
            # must : key and MindSpore requires .11/.8 
            ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            has_cublas11 = False
            has_cudnn8 = False
            for d in ld_paths:
                if not d:
                    continue
                if os.path.isfile(os.path.join(d, "libcublas.so.11")) or os.path.isfile(os.path.join(d, "libcublas.so.11.0.0")):
                    has_cublas11 = True
                if os.path.isfile(os.path.join(d, "libcudnn.so.8")) or any(os.path.isfile(os.path.join(d, x)) for x in ("libcudnn.so.8.0", "libcudnn_ops_infer.so.8")):
                    has_cudnn8 = True
            for lib in ("libcuda.so", "libcudnn.so", "libcublas.so"):
                found = []
                for d in ld_paths:
                    if not d:
                        continue
                    for f in (lib, lib + ".11", lib + ".8"):
                        if os.path.isfile(os.path.join(d, f)):
                            found.append(os.path.join(d, f))
                            break
                print(f"  {lib}: {'found ' + str(found[:2]) if found else 'NOT FOUND in LD_LIBRARY_PATH'}", file=sys.stderr)
            if not has_cublas11 or not has_cudnn8:
                print(
                    "[apply_cellfm] MindSpore 2.6 GPU must CUDA 11 (libcublas.so.11) and cuDNN 8 (libcudnn.so.8), current may CUDA 13 / cuDNN 9, .",
                    file=sys.stderr,
                )
                print(
                    " in cellfm CUDA 11 and cuDNN 8 after ( ): ",
                    file=sys.stderr,
                )
                print(
                    "    conda install -y -c nvidia cuda-toolkit=11.8 cuda-cudnn-cu11=8.9",
                    file=sys.stderr,
                )
                print(
                    " or : conda remove cuda-toolkit cudnn --force; conda install -y -c nvidia cuda-toolkit=11.8 cuda-cudnn-cu11=8.9",
                    file=sys.stderr,
                )
            print(
                "CellFM encoding on CPU is not supported (allocates too much memory).",
                file=sys.stderr,
            )
            print(
                "Please fix CUDA/cuDNN (see above) or run on a server with a compatible GPU environment.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        ms.set_context(device_target="CPU")

    from config import Config
    from model import CellFM

    geneset = _load_geneset(repo_root)
    common_genes = [g for g in adata.var_names if g in geneset]
    if len(common_genes) < 200:
        raise ValueError(
            f"Only {len(common_genes)} genes overlap with CellFM gene_info. "
            "Need at least 200. Check adata.var_names vs gene_info.csv."
        )

    adata_sub = adata[:, common_genes]
    X = adata_sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    total_counts = np.asarray(X.sum(axis=1)).ravel()
    total_counts = np.maximum(total_counts, 1.0)

    cfg = Config()
    cfg.enc_dims = 1536
    cfg.enc_nlayers = 40
    cfg.enc_num_heads = 48
    cfg.add_zero = False
    cfg.pad_zero = True
    cfg.label = False
    n_genes = len(geneset)
    model = CellFM(n_genes, cfg)
    model.set_train(False)

    param_dict = ms.load_checkpoint(ckpt_path)
    # "parameters not loaded" (MindSpore will key )
    import logging
    _ms_log = logging.getLogger("mindspore")
    _old_level = _ms_log.level
    _ms_log.setLevel(logging.ERROR)
    try:
        not_load, _ = ms.load_param_into_net(model, param_dict, strict_load=False)
    finally:
        _ms_log.setLevel(_old_level)
    if not_load:
        # Retry with common prefixes stripped (e.g. saved from wrapper)
        new_dict = {}
        for k, v in param_dict.items():
            key = k
            for prefix in ("model.", "wrapper.", "network."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            new_dict[key] = v
        _ms_log.setLevel(logging.ERROR)
        try:
            ms.load_param_into_net(model, new_dict, strict_load=False)
        finally:
            _ms_log.setLevel(_old_level)

    batch_size = 32
    n_cells = adata_sub.n_obs
    all_cls = []

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        X_b = X[start:end]
        T_b = total_counts[start:end]
        expr, gene, zero_idx = _prepare_cell_batch(
            X_b, common_genes, geneset, T_b
        )
        expr_ms = ms.Tensor(expr)
        gene_ms = ms.Tensor(gene)
        zero_idx_ms = ms.Tensor(zero_idx)
        _, _, cls_token = model.forward(expr_ms, gene_ms, zero_idx_ms)
        all_cls.append(cls_token.asnumpy())

    latent = np.concatenate(all_cls, axis=0)
    assert latent.shape == (n_cells, CELLFM_LATENT_DIM), (
        f"Expected latent shape ({n_cells}, {CELLFM_LATENT_DIM}), got {latent.shape}"
    )
    return latent


def main():
    parser = argparse.ArgumentParser(
        description="Apply CellFM encoder to AnnData and export X_cellfm."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Input AnnData .h5ad to be encoded.",
    )
    parser.add_argument(
        "--out-h5ad",
        type=str,
        required=True,
        help="Output .h5ad with adata.obsm['X_cellfm'] added.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to pretrained CellFM (MindSpore) checkpoint, e.g. CellFM_80M_weight.ckpt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' (MindSpore GPU) or 'cpu'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, overwrite existing X_cellfm in out-h5ad.",
    )
    args = parser.parse_args()

    repo_root = _get_repo_root()
    # path as for based on , cwd / 
    if os.path.isabs(args.out_h5ad):
        out_abs = args.out_h5ad
    else:
        out_abs = os.path.abspath(os.path.join(repo_root, args.out_h5ad))
    if os.path.isabs(args.data_path):
        data_abs = args.data_path
    else:
        data_abs = os.path.abspath(os.path.join(repo_root, args.data_path))

    if not os.path.isfile(data_abs):
        print(f"[apply_cellfm] ERROR: Input data not found: {data_abs}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(out_abs) and not args.force:
        print(f"[apply_cellfm] Found existing out-h5ad: {out_abs}")
        adata_existing = sc.read_h5ad(out_abs)
        if "X_cellfm" in adata_existing.obsm:
            print("[apply_cellfm] adata.obsm['X_cellfm'] already present. Skip encoding.")
            return
        print("[apply_cellfm] Out file exists but without 'X_cellfm'. Will recompute.")

    print(f"[apply_cellfm] Loading AnnData from: {data_abs}")
    adata = sc.read_h5ad(data_abs)

    device = args.device
    if device.startswith("cuda") and torch is not None and not torch.cuda.is_available():
        print("[apply_cellfm] CUDA not available, fallback to CPU.")
        device = "cpu"
    print(f"[apply_cellfm] Using device: {device}")

    try:
        print(f"[apply_cellfm] Encoding with CellFM using ckpt: {args.ckpt_path}")
        latent = encode_with_cellfm(adata, args.ckpt_path, device=device)
        print(f"[apply_cellfm] Latent shape: {latent.shape}")

        adata.obsm["X_cellfm"] = latent

        out_dir = os.path.dirname(out_abs)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        adata.write_h5ad(out_abs)
        print(f"[apply_cellfm] ✔ Saved AnnData with X_cellfm to: {out_abs}")
        print("[apply_cellfm] Done.")
    except Exception as e:
        print("[apply_cellfm] ERROR during encode or write:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
