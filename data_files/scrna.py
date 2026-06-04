# data/scrna.py
import numpy as np
import torch
from torch.utils.data import Dataset
import anndata
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.drug_structure import Drug_dose_encoder, extract_smiles_dose_from_obs


def _build_drug_dose_labels(obs, drug_key='perturbation', dose_key='dose_value', pert_status_col='perturbation_status'):
    """
    Build combined drug+dose labels for MOA task.
    Control -> 'Control', IFN -> 'drug_dose' (e.g., 'IFN-alpha_100').
    """
    pert_status = obs.get(pert_status_col, None)
    if pert_status is None:
        pert_status = pd.Series(['IFN'] * len(obs))

    drug_vals = obs[drug_key].astype(str).str.strip().replace(
        {'nan': '', 'NaN': '', 'None': '', '': 'control'}
    ) if drug_key in obs.columns else pd.Series(['control'] * len(obs))
    dose_vals = obs[dose_key].astype(float).fillna(0) if dose_key in obs.columns else pd.Series([0.0] * len(obs))

    labels = []
    for i in range(len(obs)):
        ps = str(pert_status.iloc[i]).strip()
        d = str(drug_vals.iloc[i]).strip()
        v = float(dose_vals.iloc[i])
        if ps == 'Control' or (d.lower() in ('control', '') and v == 0):
            labels.append('Control')
        else:
            labels.append(f"{d}_{int(v) if v == int(v) else v}")
    return np.array(labels)


def get_target_drug_dose_from_test(test_path, label_encoder, drug_key='perturbation', dose_key='dose_value'):
    """
    Get dominant (drug_label, dose) from test set IFN cells for MOA sampling.
    Returns (drug_label_str, drug_idx, dose_val) where drug_idx is the encoded class index.
    """
    from collections import Counter
    adata = anndata.read_h5ad(test_path)
    ifn_mask = adata.obs['perturbation_status'] == 'IFN'
    if not ifn_mask.any():
        classes = list(label_encoder.classes_)
        for i, c in enumerate(classes):
            if c != 'Control':
                return c, i, 0.0
        return 'Control', 0, 0.0

    ifn_obs = adata.obs[ifn_mask]
    labels = _build_drug_dose_labels(ifn_obs, drug_key, dose_key)
    dominant_label = Counter(labels).most_common(1)[0][0]
    if dominant_label in label_encoder.classes_:
        idx = list(label_encoder.classes_).index(dominant_label)
    else:
        classes = list(label_encoder.classes_)
        for i, c in enumerate(classes):
            if c != 'Control':
                idx = i
                dominant_label = c
                break
        else:
            idx = 0
            dominant_label = classes[0]

    dose_vals = ifn_obs[dose_key].astype(float).fillna(0) if dose_key in ifn_obs.columns else [0.0] * len(ifn_obs)
    dose_val = float(np.median(dose_vals))
    return dominant_label, idx, dose_val


def get_target_drug_emb_from_test(test_path, smiles_key='smiles', dose_key='dose_value',
                                  drug_dimension=1024):
    """
    Get dominant SMILES+dose from test IFN cells and encode with Drug_dose_encoder
    (same conditioning as Squidiff use_drug_structure=True).
    Returns (drug_emb_vector, dominant_smiles, dose_val).
    """
    from collections import Counter
    adata = anndata.read_h5ad(test_path)
    ifn_mask = adata.obs['perturbation_status'] == 'IFN'
    if not ifn_mask.any():
        smiles_list, dose_list = extract_smiles_dose_from_obs(adata.obs, smiles_key, dose_key)
        emb = Drug_dose_encoder(smiles_list[:1], dose_list[:1], num_Bits=drug_dimension)
        return emb[0], smiles_list[0], dose_list[0]

    ifn_obs = adata.obs[ifn_mask]
    smiles_list, dose_list = extract_smiles_dose_from_obs(ifn_obs, smiles_key, dose_key)
    dominant_smiles = Counter(smiles_list).most_common(1)[0][0]
    dose_val = float(np.median(dose_list))
    emb = Drug_dose_encoder([dominant_smiles], [dose_val], num_Bits=drug_dimension)
    return emb[0], dominant_smiles, dose_val


class PairedScrnaDataset(Dataset):
    """
    Builds (control vs perturbed) cell pairs from AnnData.
    Detects donor/batch columns when present; otherwise pairs globally within the pairing subset.
    scGen-style mode: if pair_only_obs_key / pair_only_obs_value are set, pairs are built only
    among cells matching that obs filter (e.g. split=='train'); other cells are not used for pairing.
    """
    def __init__(self, adata_path, donor_key=None, ctrl_status='Control', pert_status='IFN',
                 pair_only_obs_key=None, pair_only_obs_value=None):
        """
        Args:
            adata_path: Path to .h5ad file.
            donor_key: Optional column for stratified pairing (e.g. 'donor', 'batch').
            ctrl_status: Value in ``perturbation_status`` for control cells.
            pert_status: Value in ``perturbation_status`` for perturbed cells.
            pair_only_obs_key: If set with pair_only_obs_value, restrict pairing to cells with
                obs[key]==value (e.g. scGen: only ``split=='train'``).
            pair_only_obs_value: See pair_only_obs_key.
        """
        adata = anndata.read_h5ad(adata_path)
        obs = adata.obs.copy()
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        # scGen-style: pair only within cells matching pair_only filter
        if pair_only_obs_key is not None and pair_only_obs_value is not None and pair_only_obs_key in obs.columns:
            obs_for_pair = obs[obs[pair_only_obs_key].astype(str) == str(pair_only_obs_value)]
            print(
                f"INFO: scGen pair-only mode: pairing only cells with "
                f"obs['{pair_only_obs_key}']=='{pair_only_obs_value}' (n={len(obs_for_pair)})."
            )
        else:
            obs_for_pair = obs

        self.pairs = []
        
        # --- Pairing logic (on obs_for_pair) ---
        
        donor_key_found = None
        
        if donor_key_found:
            print(f"INFO: pairing within groups of column '{donor_key_found}'...")
            for group_id, sub_obs in obs_for_pair.groupby(donor_key_found):
                idx_ctrl = sub_obs[sub_obs['perturbation_status'] == ctrl_status].index
                idx_pert = sub_obs[sub_obs['perturbation_status'] == pert_status].index
                if len(idx_ctrl) > 0 and len(idx_pert) > 0:
                    n = min(len(idx_ctrl), len(idx_pert))
                    for i in range(n):
                        self.pairs.append((idx_ctrl[i], idx_pert[i]))
        else:
            # Strategy B: global pairing within obs_for_pair
            if pair_only_obs_key is None:
                print("INFO: no donor/batch key; pairing globally within the cohort.")
            idx_ctrl = obs_for_pair[obs_for_pair['perturbation_status'] == ctrl_status].index
            idx_pert = obs_for_pair[obs_for_pair['perturbation_status'] == pert_status].index
            if len(idx_ctrl) > 0 and len(idx_pert) > 0:
                n = min(len(idx_ctrl), len(idx_pert))
                for i in range(n):
                    self.pairs.append((idx_ctrl[i], idx_pert[i]))

        # --- End pairing logic ---

        if not self.pairs:
            print(
                f"\nERROR: no paired samples. Check that 'perturbation_status' contains "
                f"'{ctrl_status}' and '{pert_status}'."
            )
        else:
            print(f"\nBuilt {len(self.pairs)} control–perturbation pairs.")

        self.X = X
        # obs index for fast __getitem__ lookups
        self.obs = obs.set_index(pd.Index(obs.index))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        i0, i1 = self.pairs[i]
        v0 = self.X[self.obs.index.get_loc(i0)]
        v1 = self.X[self.obs.index.get_loc(i1)]
        return torch.from_numpy(v0).float(), torch.from_numpy(v1).float()

class PairedScrnaDatasetDrugCond(Dataset):
    """
    Paired (Control, IFN) dataset with drug conditioning for MOA task.

    use_drug_structure=False (default): returns (ctrl, pert, drug_idx, dose)
    use_drug_structure=True (Squidiff-style): returns (ctrl, pert, drug_emb)
        where drug_emb is SMILES+dose Morgan fingerprint (1024-dim).
    """
    def __init__(self, adata_path, drug_key='perturbation', dose_key='dose_value',
                 smiles_key='smiles', ctrl_status='Control', pert_status='IFN',
                 use_drug_structure=False, drug_dimension=1024):
        adata = anndata.read_h5ad(adata_path)
        obs = adata.obs.copy()
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        self.pairs = []
        idx_ctrl = obs[obs['perturbation_status'] == ctrl_status].index
        idx_pert = obs[obs['perturbation_status'] == pert_status].index

        if len(idx_ctrl) > 0 and len(idx_pert) > 0:
            n = min(len(idx_ctrl), len(idx_pert))
            for i in range(n):
                self.pairs.append((idx_ctrl[i], idx_pert[i]))

        if not self.pairs:
            raise ValueError(
                f"No paired samples. Check perturbation_status for '{ctrl_status}' and '{pert_status}'."
            )

        self.use_drug_structure = use_drug_structure
        mode = "SMILES+dose" if use_drug_structure else "drug_name+dose"
        print(f"PairedScrnaDatasetDrugCond ({mode}): {len(self.pairs)} pairs from {adata_path}")

        self.X = X
        self.obs = obs.set_index(obs.index)
        self.drug_key = drug_key
        self.dose_key = dose_key
        self.smiles_key = smiles_key
        self.drug_dimension = drug_dimension

        labels = _build_drug_dose_labels(obs, drug_key, dose_key)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.label_indices = self.label_encoder.transform(labels)
        self.dose_values = obs[dose_key].astype(float).fillna(0).values if dose_key in obs.columns else np.zeros(len(obs))

        if use_drug_structure:
            smiles_list, dose_list = extract_smiles_dose_from_obs(obs, smiles_key, dose_key)
            drug_emb = Drug_dose_encoder(smiles_list, dose_list, num_Bits=drug_dimension)
            self.drug_emb = drug_emb.astype(np.float32)

    def get_label_encoder(self):
        return self.label_encoder

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        i0, i1 = self.pairs[i]
        loc0 = self.obs.index.get_loc(i0)
        loc1 = self.obs.index.get_loc(i1)
        v0 = self.X[loc0]
        v1 = self.X[loc1]
        if self.use_drug_structure:
            drug_emb = torch.from_numpy(self.drug_emb[loc1]).float()
            return (
                torch.from_numpy(v0).float(),
                torch.from_numpy(v1).float(),
                drug_emb,
            )
        drug_idx = self.label_indices[loc1]
        dose = float(self.dose_values[loc1])
        return (
            torch.from_numpy(v0).float(),
            torch.from_numpy(v1).float(),
            drug_idx,
            dose,
        )


class EmbeddingDataset(Dataset):
    def __init__(self, npy_path: str):
        """
        Args:
            npy_path (str): Path to the .npy file containing embeddings.
        """
        self.embeddings = np.load(npy_path)
        print(f"Loaded embeddings from {npy_path} with shape {self.embeddings.shape}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Return the embedding as a torch tensor
        return torch.from_numpy(self.embeddings[idx]).float()