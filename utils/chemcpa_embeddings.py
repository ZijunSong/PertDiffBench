"""On-the-fly drug embeddings for ChemCPA (Morgan FCFP, SMILES + dose via doser)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.drug_structure import split_smiles_advanced


def _fp_from_smiles(smiles: str, num_bits: int = 2048) -> np.ndarray:
    parts = split_smiles_advanced(smiles)
    combined = np.zeros(num_bits, dtype=np.float32)
    valid = 0
    for smi in parts:
        smi = smi.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_bits)
        combined += np.array(list(fp.ToBitString()), dtype=np.float32)
        valid += 1
    if valid > 1:
        combined /= valid
    return combined


def smiles_to_morgan_fp(smiles: Optional[str], num_bits: int = 2048) -> np.ndarray:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return np.zeros(num_bits, dtype=np.float32)
    text = str(smiles).strip()
    if not text or text.lower() in {"nan", "none", "null", ""}:
        return np.zeros(num_bits, dtype=np.float32)
    return _fp_from_smiles(text, num_bits=num_bits)


def build_embedding_matrix(
    smiles_list: Iterable[Optional[str]],
    num_bits: int = 2048,
) -> tuple[list[str], np.ndarray]:
    keys: list[str] = []
    rows: list[np.ndarray] = []
    for i, smi in enumerate(smiles_list):
        key = str(smi) if smi not in (None, "") else f"__missing_{i}__"
        keys.append(key)
        rows.append(smiles_to_morgan_fp(smi, num_bits=num_bits))
    return keys, np.stack(rows, axis=0)


def save_drug_embeddings_parquet(
    smiles_list: Iterable[Optional[str]],
    output_path: str | Path,
    num_bits: int = 2048,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys, mat = build_embedding_matrix(smiles_list, num_bits=num_bits)
    df = pd.DataFrame(
        mat,
        index=keys,
        columns=[f"fp_{i}" for i in range(mat.shape[1])],
    )
    if df.index.duplicated().any():
        n_dup = int(df.index.duplicated().sum())
        print(f"Deduplicating {n_dup} duplicate SMILES keys in drug embeddings")
        df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(output_path)
    print(f"Saved ChemCPA drug embeddings: {output_path} (shape={df.shape})")
    return output_path
