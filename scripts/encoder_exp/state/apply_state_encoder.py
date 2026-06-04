#!/usr/bin/env python3
"""
Apply pretrained State Embedding (SE) model to an AnnData file and write X_state.

Uses State Python API directly to reuse model instance and avoid OOM.
This avoids reloading the model for each chunk (which causes return code -9).

Install: uv tool install arc-state
"""

import os
import sys
import tempfile
import argparse
import shutil

import numpy as np
import scanpy as sc

# State package path
STATE_PKG_PATH = os.path.expanduser("~/.local/share/uv/tools/arc-state/lib/python3.10/site-packages")
STATE_PYTHON = os.path.expanduser("~/.local/share/uv/tools/arc-state/bin/python")

# Check if we should use State's Python interpreter
USE_STATE_PYTHON = os.path.exists(STATE_PYTHON) and os.path.exists(STATE_PKG_PATH)

if USE_STATE_PYTHON:
    # Add State package to path
    sys.path.insert(0, STATE_PKG_PATH)
    sys.path.insert(0, os.path.dirname(STATE_PKG_PATH))


def main():
    # Import State modules - use State's Python environment if available
    try:
        from state.emb.inference import Inference
        from omegaconf import OmegaConf
        import torch
    except ImportError as e:
        if USE_STATE_PYTHON:
            # If State Python exists but import fails, suggest using it directly
            print(f"[ERROR] Failed to import State Python API: {e}", file=sys.stderr)
            print(f"[ERROR] State package path: {STATE_PKG_PATH}", file=sys.stderr)
            print(f"[ERROR] Try running with State's Python interpreter:", file=sys.stderr)
            print(f"[ERROR]   {STATE_PYTHON} {__file__} [args...]", file=sys.stderr)
        else:
            print(f"[ERROR] Failed to import State Python API: {e}", file=sys.stderr)
            print(f"[ERROR] State Python not found at: {STATE_PYTHON}", file=sys.stderr)
            print(f"[ERROR] Ensure 'uv tool install arc-state' completed successfully.", file=sys.stderr)
        sys.exit(1)

    # cuDNN-backed scaled_dot_product_attention can fail on some GPUs / driver stacks
    # with: RuntimeError: cuDNN Frontend error: ... No valid execution plans built.
    # Disabling it forces Flash / mem_efficient / math SDPA instead (slower but stable).
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
    
    parser = argparse.ArgumentParser(
        description="Apply pretrained State Embedding (SE) to AnnData and export X_state."
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
        help="Output .h5ad with adata.obsm['X_state'] added.",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        required=True,
        help="Directory of State SE model (e.g., SE-600M from HuggingFace).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .ckpt. If None, uses model-folder/*.ckpt.",
    )
    parser.add_argument(
        "--batch-cells",
        type=int,
        default=500,
        help="Process cells in chunks of this size to reduce memory (default: 500). "
             "Decrease if OOM (Killed) occurs. Smaller chunks use less memory but are slower.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding forward pass (default: 32). "
             "Smaller values use less VRAM/RAM. Try 16 or 8 if OOM.",
    )
    args = parser.parse_args()

    model_folder = os.path.abspath(args.model_folder)
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    # Resolve checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        import glob
        candidates = sorted(glob.glob(os.path.join(model_folder, "*.ckpt")))
        if not candidates:
            raise FileNotFoundError(
                f"No .ckpt file found in {model_folder}. "
                "Please specify --checkpoint explicitly."
            )
        ckpt = candidates[-1]  # Use latest checkpoint
        print(f"[apply_state] Auto-selected checkpoint: {ckpt}")
    ckpt = os.path.abspath(ckpt)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    data_path = os.path.abspath(args.data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Input data not found: {data_path}")

    print(f"[apply_state] Loading AnnData from: {data_path}")
    adata = sc.read_h5ad(data_path)
    n_cells = adata.n_obs
    batch_size = min(args.batch_cells, n_cells)

    # Resolve protein embeddings (same logic as CLI)
    protein_embeds = None
    if os.path.exists(os.path.join(model_folder, "protein_embeddings.pt")):
        pe_path = os.path.join(model_folder, "protein_embeddings.pt")
        print(f"[apply_state] Found protein embeddings: {pe_path}")
        protein_embeds = torch.load(pe_path, weights_only=False, map_location="cpu")

    # Initialize Inference object and load model ONCE
    print(f"[apply_state] Loading State SE model from: {ckpt}")
    print("[apply_state] This may take a moment and use significant memory...")
    print(f"[apply_state] Using chunk size: {batch_size} cells, embed batch size: {args.embed_batch_size}")
    
    # Free adata memory before loading model (we'll reload chunks as needed)
    del adata
    import gc
    gc.collect()
    
    inferer = Inference(cfg=None, protein_embeds=protein_embeds)
    inferer.load_model(ckpt)
    print("[apply_state] Model loaded successfully. Processing cells...")
    
    # Reload adata for chunking
    adata = sc.read_h5ad(data_path)

    # Process in chunks to avoid OOM
    all_embeddings = []
    n_chunks = (n_cells + batch_size - 1) // batch_size
    print(f"[apply_state] Processing {n_cells} cells in {n_chunks} chunk(s) of up to {batch_size} cells each")

    tmpdir = tempfile.mkdtemp(prefix="state_apply_")
    try:
        for i in range(n_chunks):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_cells)
            chunk = adata[start:end]
            chunk_h5ad = os.path.join(tmpdir, f"chunk_{i}.h5ad")
            chunk_npy = os.path.join(tmpdir, f"chunk_{i}.npy")

            chunk.write_h5ad(chunk_h5ad)
            print(f"[apply_state] Chunk {i+1}/{n_chunks}: cells {start}-{end}")

            # Use Python API: encode_adata returns embeddings directly
            # Model is reused from the Inference instance
            embeddings_chunk = inferer.encode_adata(
                input_adata_path=chunk_h5ad,
                output_adata_path=None,  # We'll save manually
                emb_key="X_state",
                batch_size=args.embed_batch_size,  # Forward pass batch size
            )

            if embeddings_chunk is None:
                raise RuntimeError(f"Failed to generate embeddings for chunk {i+1}")

            # Save chunk embeddings
            np.save(chunk_npy, embeddings_chunk)
            all_embeddings.append(embeddings_chunk)

            # Free memory explicitly
            del embeddings_chunk
            os.remove(chunk_h5ad)
            os.remove(chunk_npy)
            import gc
            gc.collect()

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        print(f"[apply_state] Embeddings shape: {embeddings.shape}")
        adata.obsm["X_state"] = embeddings
    finally:
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)

    out_dir = os.path.dirname(args.out_h5ad)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    adata.write_h5ad(args.out_h5ad)
    print(f"[apply_state] Saved AnnData with X_state to: {args.out_h5ad}")
    print("[apply_state] Done.")


if __name__ == "__main__":
    main()
