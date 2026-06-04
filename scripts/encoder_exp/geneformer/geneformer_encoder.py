import os
import pathlib
from typing import Optional, Dict

import numpy as np
import anndata as ad

from geneformer import TranscriptomeTokenizer, EmbExtractor
import inspect


class GeneformerEncoder:
    """
    High-level wrapper around Geneformer tokenizer + EmbExtractor.

    This class:
    1) Tokenizes raw scRNA h5ad to Geneformer .dataset format (if not already present).
    2) Extracts cell embeddings using the pretrained Geneformer model (if not already present).
    3) Writes embeddings back into an AnnData .h5ad file in obsm["X_geneformer"].
    """

    def __init__(
        self,
        geneformer_root: str,
        model_version: str = "V2",  # just a tag for later; not all versions support this arg
        nproc: int = 8,
    ):
        """
        Args:
            geneformer_root: Path to the cloned Geneformer repo (where config.json & model.safetensors live).
            model_version: "V2" or "V1" - will be used only if your installed Geneformer supports it.
            nproc: Number of processes for tokenizer / embedding extractor.
        """
        self.geneformer_root = pathlib.Path(geneformer_root)
        self.model_version = model_version
        self.nproc = nproc

        if not self.geneformer_root.exists():
            raise FileNotFoundError(
                f"Geneformer root '{self.geneformer_root}' not found. "
                f"Clone from https://huggingface.co/ctheodoris/Geneformer first."
            )

    @staticmethod
    def _ensure_dir(path: pathlib.Path):
        path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Tokenize
    # ------------------------------------------------------------------
    def tokenize_h5ad(
        self,
        input_h5ad: str,
        output_dir: str,
        output_prefix: str,
        custom_attr_name_dict: Optional[Dict[str, str]] = None,
        resume: bool = True,
    ) -> str:
        """
        Tokenize raw counts h5ad into Geneformer .dataset format.

        as Geneformer directory logic, here : 
        1) no longerdirectlyoriginaldirectorypass to Geneformer; 
        2) in output_dir under tempsubdir, copycurrentthis one h5ad into; 
        3) tempsubdir as data_directory tokenize_data, willhandlethis onefile.
        """
        output_dir = pathlib.Path(output_dir)
        self._ensure_dir(output_dir)
        dataset_path = output_dir / f"{output_prefix}.dataset"

        if dataset_path.exists() and resume:
            print(f"[GeneformerEncoder] Found existing dataset: {dataset_path}, skip tokenization.")
            return str(dataset_path)

        print(f"[GeneformerEncoder] Tokenizing {input_h5ad} -> {dataset_path}")

        # 1) prepare TranscriptomeTokenizer
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict if custom_attr_name_dict is not None else {},
            nproc=self.nproc,
        )

        # 2) build tempdirectory, h5ad
        import shutil
        input_path = pathlib.Path(input_h5ad)
        if not input_path.exists():
            raise FileNotFoundError(f"Input h5ad not found: {input_h5ad}")

        tmp_dir = output_dir / f"_tmp_tokenize_{output_prefix}"
        self._ensure_dir(tmp_dir)

        tmp_h5ad = tmp_dir / input_path.name
        # as , under, prepare 
        shutil.copy2(input_path, tmp_h5ad)

        data_directory = str(tmp_dir)
        print(f"[GeneformerEncoder] Using temporary data_directory={data_directory} for tokenize_data")

        # 3) based on Geneformer whether model_version
        sig = inspect.signature(tk.tokenize_data)
        kwargs = dict(
            data_directory=data_directory,
            output_directory=str(output_dir),
            output_prefix=output_prefix,
            file_format="h5ad",
        )
        if "model_version" in sig.parameters:
            kwargs["model_version"] = self.model_version
            print(f"[GeneformerEncoder] tokenize_data supports 'model_version', using '{self.model_version}'.")
        else:
            print("[GeneformerEncoder] tokenize_data does NOT support 'model_version'; "
                  "calling without it (older Geneformer version).")

        # 4) actualcall Geneformer  tokenize_data
        tk.tokenize_data(**kwargs)

        if not dataset_path.exists():
            raise RuntimeError(
                f"[GeneformerEncoder] Tokenization completed but .dataset not found at {dataset_path}. "
                f"Check tokenizer output and paths."
            )

        print(f"[GeneformerEncoder] Tokenized dataset saved at: {dataset_path}")
        # tempdirectory , debug; must canto or shutil.rmtree(tmp_dir)
        return str(dataset_path)


    # ------------------------------------------------------------------
    # 2) Extract embeddings
    # ------------------------------------------------------------------
    def extract_embeddings(
        self,
        dataset_path: str,
        output_dir: str,
        output_prefix: str,
        model_dir: Optional[str] = None,
        emb_mode: str = "cell",  # cell-wise pooled embeddings
        emb_layer: int = -1,
        max_ncells: Optional[int] = None,
        resume: bool = True,
    ) -> str:
        """
        Extract cell embeddings from Geneformer.

        Args:
            dataset_path: Path to tokenized .dataset file.
            output_dir: Directory where embeddings will be saved.
            output_prefix: Prefix for output embedding file.
            model_dir: Path to Geneformer model checkpoint dir; if None, use geneformer_root.
            emb_mode: "cell" (mean pooled over tokens) or "cls".
            emb_layer: -1 -> 2nd last, 0 -> last layer.
            max_ncells: If given, subsample; None = all cells.
            resume: If True, reuse existing embeddings CSV if present.

        Returns:
            Path to embeddings CSV (cell-wise embeddings).
        """
        output_dir = pathlib.Path(output_dir)
        self._ensure_dir(output_dir)

        emb_csv = output_dir / f"{output_prefix}_embs.csv"
        emb_pt = output_dir / f"{output_prefix}_embs.pt"

        if emb_csv.exists() and resume:
            print(f"[GeneformerEncoder] Found existing embeddings: {emb_csv}, skip extraction.")
            return str(emb_csv)

        model_dir = pathlib.Path(model_dir) if model_dir is not None else self.geneformer_root
        if not model_dir.exists():
            raise FileNotFoundError(f"model_dir {model_dir} not found")

        print(f"[GeneformerEncoder] Extracting embeddings from {dataset_path} using model at {model_dir}")

        # --- EmbExtractor name ---
        emb_sig = inspect.signature(EmbExtractor.__init__)
        emb_kwargs = dict(
            model_type="Pretrained",
            num_classes=0,
            emb_mode=emb_mode,
            max_ncells=max_ncells,
            emb_layer=emb_layer,
            emb_label=None,
            labels_to_plot=None,
            forward_batch_size=8,
            nproc=self.nproc,
        )
        if "model_version" in emb_sig.parameters:
            emb_kwargs["model_version"] = self.model_version
            print(f"[GeneformerEncoder] EmbExtractor supports 'model_version', using '{self.model_version}'.")
        else:
            print("[GeneformerEncoder] EmbExtractor does NOT support 'model_version'; "
                  "calling without it (older Geneformer version).")

        embex = EmbExtractor(**emb_kwargs)

        embs_df = embex.extract_embs(
            str(model_dir),
            dataset_path,
            str(output_dir),
            output_prefix,
        )

        embs_df.to_csv(emb_csv, index=False)
        try:
            import torch
            # directly df numpy , after must .pt 
            torch.save(embs_df.to_numpy(), emb_pt)
        except ImportError:
            print("[GeneformerEncoder] torch not available, skip saving .pt file")

        print(f"[GeneformerEncoder] Embeddings saved at: {emb_csv}")
        return str(emb_csv)

    # ------------------------------------------------------------------
    # 3) h5ad.obsm
    # ------------------------------------------------------------------
    def write_embeddings_to_h5ad(
        self,
        input_h5ad: str,
        emb_csv: str,
        output_h5ad: str,
        obsm_key: str = "X_geneformer",
        resume: bool = True,
    ) -> str:
        """
        Merge embeddings into AnnData.obsm and save.

        Assumes the order of cells in the .dataset / embeddings matches input_h5ad.
        For standard Geneformer tokenization of a single h5ad, this holds.

        Args:
            input_h5ad: Original h5ad path.
            emb_csv: CSV file produced by extract_embeddings.
            output_h5ad: Path to write new h5ad with obsm[obsm_key].
            obsm_key: Key name for embeddings in obsm.
            resume: If True and output_h5ad exists, skip.

        Returns:
            Path to output_h5ad.
        """
        output_path = pathlib.Path(output_h5ad)
        if output_path.exists() and resume:
            print(f"[GeneformerEncoder] Found existing encoded h5ad: {output_path}, skip writing.")
            return str(output_path)

        print(f"[GeneformerEncoder] Writing embeddings into h5ad: {output_path}")
        adata = ad.read_h5ad(input_h5ad)

        import pandas as pd
        embs_df = pd.read_csv(emb_csv)
        embs = embs_df.to_numpy(dtype=np.float32)

        if embs.shape[0] != adata.n_obs:
            raise ValueError(
                f"Mismatch in cell number: embeddings have {embs.shape[0]} cells, "
                f"adata has {adata.n_obs}. You may need to check filtering / ordering."
            )

        adata.obsm[obsm_key] = embs
        adata.write_h5ad(output_path)
        print(f"[GeneformerEncoder] Saved encoded h5ad with obsm['{obsm_key}'] at {output_path}")
        return str(output_path)
