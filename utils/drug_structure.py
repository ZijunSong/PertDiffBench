"""SMILES + dose encoding (same as Squidiff Drug_dose_encoder)."""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def split_smiles_advanced(smiles):
    """Split SMILES on top-level ';' while preserving bracketed content."""
    if ';' not in smiles:
        return [smiles]

    parts = []
    current = ""
    bracket_depth = 0

    for char in smiles:
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == ';' and bracket_depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
            continue
        current += char

    if current.strip():
        parts.append(current.strip())

    return parts if parts else [smiles]


def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    Encode SMILES to rFCFP fingerprint and scale by log10(dose + 1).
    Adopted from Squidiff / PRnet.
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits), dtype=np.float32)

    for i, smiles in enumerate(drug_SMILES_list):
        if not smiles or smiles == '' or pd.isna(smiles):
            continue

        smiles_parts = split_smiles_advanced(smiles)
        combined_fingerprint = np.zeros(num_Bits, dtype=np.float32)
        valid_parts = 0

        for smi in smiles_parts:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    print(f"Warning: Cannot parse SMILES '{smi}' (part of '{smiles}')")
                    continue
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 2, useFeatures=True, nBits=num_Bits
                ).ToBitString()
                combined_fingerprint += np.array(list(fcfp4), dtype=np.float32)
                valid_parts += 1
            except Exception as e:
                print(f"Warning: Error processing SMILES part '{smi}': {e}")
                continue

        if valid_parts > 0:
            try:
                dose_val = float(dose_list[i]) if dose_list[i] not in ['', None] else 0.0
                if dose_val > 0:
                    combined_fingerprint = combined_fingerprint * np.log10(dose_val + 1)
                fcfp4_array[i] = combined_fingerprint
            except (ValueError, TypeError):
                print(f"Warning: Invalid dose value at index {i}: {dose_list[i]}")
                fcfp4_array[i] = combined_fingerprint

    return fcfp4_array


def extract_smiles_dose_from_obs(obs, smiles_key='smiles', dose_key='dose_value'):
    """Return (smiles_list, dose_float_list) from AnnData obs."""
    if smiles_key not in obs.columns:
        raise ValueError(f"obs missing '{smiles_key}' column (required for use_drug_structure=True)")

    smiles_series = obs[smiles_key].astype(str).replace({
        'nan': '', 'NaN': '', 'None': '', 'null': ''
    })
    if dose_key in obs.columns:
        dose_series = obs[dose_key].astype(str).replace({
            'nan': '0', 'NaN': '0', 'None': '0', 'null': '0'
        })
        dose_list = [float(x) if x != '' else 0.0 for x in dose_series.to_list()]
    else:
        dose_list = [0.0] * len(obs)

    return smiles_series.to_list(), dose_list
