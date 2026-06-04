# utils/scgpt_utils.py

import torch
import numpy as np
from typing import List, Tuple
from tdc.model_server.tokenizers.scgpt import scGPTTokenizer
from scgpt.tokenizer.gene_tokenizer import (
    get_default_gene_vocab,
    tokenize_and_pad_batch
)

def load_scgpt(cfg, device):
    from tdc import tdc_hf_interface
    scgpt = tdc_hf_interface(cfg.model.scgpt.model_name)
    model = scgpt.load().to(device)
    tokenizer = scGPTTokenizer()
    return model, tokenizer

def _filter_to_vocab(
    counts: np.ndarray,
    gene_names: List[str],
    stoi: dict
) -> Tuple[np.ndarray, List[str]]:
    """
    Keep only genes present in the vocabulary (stoi).

    - counts: raw expression matrix, shape [B, G]
    - gene_names: original gene name list, length G
    - stoi: gene -> index mapping

    Returns:
    - new_counts: matrix with valid columns only, shape [B, G']
    - new_gene_names: valid gene names, length G'
    """
    valid_pairs = [(i, g) for i, g in enumerate(gene_names) if g in stoi]
    if not valid_pairs:
        raise ValueError("No genes in vocabulary; check input data and vocab match.")
    valid_indices, valid_genes = zip(*valid_pairs)
    valid_indices = list(valid_indices)
    valid_genes = list(valid_genes)

    new_counts = counts[:, valid_indices]
    return new_counts, valid_genes

def embed_cells(
    model: torch.nn.Module,
    tokenizer: scGPTTokenizer,
    x: torch.Tensor,
    gene_names: List[str],
) -> torch.Tensor:
    """
    1) Map gene names to integer indices in vocab (gene_ids_array, np.int64, length G);
    2) Call tokenize_and_pad_batch for tokenize + padding;
    3) Return CLS token embedding.
    """
    print(f"[embed_cells DEBUG] input x  shape: {x.shape}")
    print(f"[embed_cells DEBUG] input gene_names length: {len(gene_names)}")
    if gene_names:
        print(f"[embed_cells DEBUG] gene_names example (first 5): {gene_names[:5]}")

    device = model.device
    counts = x.detach().cpu().numpy()
    B, G = counts.shape

    print(f"[embed_cells DEBUG] counts type: {type(counts)}, dtype: {counts.dtype}")
    print(f"[embed_cells DEBUG] counts[0,:5] example: {counts[0, :5]}")

    # 1) Prepare vocab and register special tokens
    vocab = get_default_gene_vocab()
    stoi = vocab.get_stoi()
    print(f"[embed_cells DEBUG] default vocab example (first 5 stoi): {list(vocab.get_stoi().items())[:5]}")
    print(f"[embed_cells DEBUG] default vocab size: {len(vocab)}")

    for special_token in ("<pad>", "<cls>"):
        if special_token not in vocab.get_stoi():
            stoi[special_token] = max(stoi.values()) + 1 if stoi else 0
            vocab = vocab.from_dict(stoi)
            print(f"[embed_cells DEBUG] added special token '{special_token}' to vocab")
    vocab.set_default_token(special_token)
    stoi = vocab.get_stoi()
    print(f"[embed_cells DEBUG] updated vocab example (first 5 stoi): {list(vocab.get_stoi().items())[:5]}")
    print(f"[embed_cells DEBUG] updated vocab size: {len(vocab)}")

    # 2) Filter to intersection with vocab
    counts, gene_names = _filter_to_vocab(counts, gene_names, stoi)
    B, G = counts.shape
    print(f"[embed_cells DEBUG] after filter G'={G}, kept genes example: {gene_names[:5]}")

    counts = np.log1p(counts)
    min_vals = counts.min(axis=0, keepdims=True)
    max_vals = counts.max(axis=0, keepdims=True)
    counts = (counts - min_vals) / (max_vals - min_vals + 1e-6)
    print(f"[embed_cells DEBUG] normalized counts: min={counts.min():.4f}, max={counts.max():.4f}")

    gene_ids_array = np.array([stoi[g] for g in gene_names], dtype=np.int64)
    print(f"[embed_cells DEBUG] gene_ids_array shape: {gene_ids_array.shape}")
    print(f"[embed_cells DEBUG] gene_ids_array dtype: {gene_ids_array.dtype}")
    print(f"[embed_cells DEBUG] gene_ids_array example (first 5): {gene_ids_array[:5]}")

    assert gene_ids_array.shape[0] == G, (
        f"Gene count G ({G}) and gene_ids_array length ({gene_ids_array.shape[0]}) mismatch; "
        "some gene names may be missing from the vocabulary."
    )

    print("[embed_cells DEBUG] calling tokenize_and_pad_batch...")
    print(f"[embed_cells DEBUG] data (counts) shape: {counts.shape}, type: {type(counts)}")
    print(f"[embed_cells DEBUG] gene_ids shape: {gene_ids_array.shape}, dtype: {gene_ids_array.dtype}")
    print(f"[embed_cells DEBUG]   max_len: {G}")
    print(f"[embed_cells DEBUG]   vocab object: {type(vocab)}")
    print(f"[embed_cells DEBUG]   pad_token: '<pad>'")
    print(f"[embed_cells DEBUG]   cls_token: '<cls>'")
    print(f"[embed_cells DEBUG]   pad_value: 0")
    print(f"[embed_cells DEBUG]   append_cls: True")
    print(f"[embed_cells DEBUG]   return_pt: True")

    try:
        batch = tokenize_and_pad_batch(
            data=counts,
            gene_ids=gene_ids_array,
            max_len=G,
            vocab=vocab,
            pad_token="<pad>",
            cls_token="<cls>",
            pad_value=0,
            append_cls=True,
            return_pt=True,
        )
    except TypeError as e:
        print(f"[embed_cells ERROR] tokenize_and_pad_batch TypeError: {e}")
        print("[embed_cells DEBUG HINT] check tokenize_batch inputs: gene_ids_array and idx.")
        print("[embed_cells DEBUG HINT] gene_ids_array should be 1D NumPy int64.")
        print("[embed_cells DEBUG HINT] idx must be integer scalar or valid integer indices.")
        raise e

    print(f"[embed_cells DEBUG] tokenize_and_pad_batch returned keys: {batch.keys()}")
    print(f"[embed_cells DEBUG] batch['genes'] shape: {batch['genes'].shape}, dtype: {batch['genes'].dtype}")
    print(f"[embed_cells DEBUG] batch['values'] shape: {batch['values'].shape}, dtype: {batch['values'].dtype}")

    input_ids = batch["genes"].to(device)
    values = batch["values"].to(device)
    pad_id = stoi["<pad>"]

    num_total = values.numel()
    nan_count = torch.isnan(values).sum().item()
    nan_ratio = nan_count / num_total
    print(f"[embed_cells DEBUG] values NaN count: {nan_count}/{num_total} ({nan_ratio:.2%})")

    nan_per_row = torch.isnan(values).sum(dim=1)
    rows_with_nan = (nan_per_row > 0).nonzero(as_tuple=False).view(-1).tolist()
    print(f"[embed_cells DEBUG] rows with NaN (first 10): {rows_with_nan[:10]} (total {len(rows_with_nan)})")
    print(f"[embed_cells DEBUG] NaN per affected row (first 10): {nan_per_row[rows_with_nan[:10]].tolist()}")

    nan0 = torch.isnan(values).sum().item()
    if nan0 > 0:
        print(f"[embed_cells DEBUG] replacing {nan0} NaN values with 0")
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    pad_mask = (input_ids == pad_id)
    if pad_mask.any():
        values = values.masked_fill(pad_mask, 0.0)

    assert not torch.isnan(values).any(), "values still contain NaN"
    assert not torch.isnan(input_ids.float()).any(), "input_ids contain NaN after float cast"

    attention_mask = (input_ids != pad_id)

    print(f"[embed_cells DEBUG] scGPT input_ids shape: {input_ids.shape}")
    print(f"[embed_cells DEBUG] scGPT values shape: {values.shape}")

    with torch.autograd.set_detect_anomaly(True):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=values
        )
    print(outputs)

    print(f"[embed_cells DEBUG] model output keys: {list(outputs.keys())}")

    if "cell_emb" in outputs:
        cls_embedding = outputs["cell_emb"]
    elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
        cls_embedding = outputs["last_hidden_state"][:, 0, :]
    else:
        hidden = outputs.last_hidden_state
        cls_embedding = hidden[:, 0, :]

    print(f"[embed_cells DEBUG] CLS embedding shape: {cls_embedding.shape}")
    return cls_embedding
