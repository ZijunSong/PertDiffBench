import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import scanpy as sc
import torch
import sys
import scipy
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder
import time


def _build_drug_dose_labels(adata, drug_key='perturbation', dose_key='dose_value'):
    """Build combined drug+dose labels for MOA task. Control -> 'Control', IFN -> 'drug_dose'."""
    pert_status = adata.obs.get('perturbation_status', None)
    if pert_status is None:
        pert_status = np.array(['IFN'] * adata.n_obs)

    drug_vals = adata.obs[drug_key].astype(str).str.strip().replace(
        {'nan': '', 'NaN': '', 'None': '', '': 'control'}
    ) if drug_key in adata.obs else pd.Series(['control'] * adata.n_obs)
    dose_vals = adata.obs[dose_key].astype(float).fillna(0) if dose_key in adata.obs else pd.Series([0.0] * adata.n_obs)

    labels = []
    for i in range(adata.n_obs):
        ps = str(pert_status.iloc[i]).strip() if hasattr(pert_status, 'iloc') else str(pert_status[i]).strip()
        d = str(drug_vals.iloc[i]).strip() if hasattr(drug_vals, 'iloc') else str(drug_vals[i]).strip()
        v = float(dose_vals.iloc[i]) if hasattr(dose_vals, 'iloc') else float(dose_vals[i])
        if ps == 'Control' or (d.lower() in ('control', '') and v == 0):
            labels.append('Control')
        else:
            labels.append(f"{d}_{int(v) if v == int(v) else v}")
    return np.array(labels)


def get_label_encoder_and_num_class(
    data_dir,
    use_drug_cond=False,
    drug_key='perturbation',
    dose_key='dose_value',
    model_path=None,
    label_key='perturbation_status',
):
    """
    Build label encoder from h5ad and optionally save. Returns (label_encoder, num_class).
    For classifier_train: use this to get num_class before creating model, and save encoder.
    When label_key is given (e.g. 'treatment_time' for fig4 time-conditioned), use that obs column.
    """
    adata = sc.read_h5ad(data_dir)
    adata.var_names_make_unique()

    if use_drug_cond and drug_key in adata.obs and dose_key in adata.obs:
        labels = _build_drug_dose_labels(adata, drug_key, dose_key)
    elif label_key and label_key in adata.obs:
        labels = np.array([str(x).strip() for x in adata.obs[label_key].values])
    else:
        labels = adata.obs['perturbation_status'].values

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    num_class = len(label_encoder.classes_)

    if model_path is not None and (use_drug_cond or (label_key and label_key != 'perturbation_status')):
        os.makedirs(model_path, exist_ok=True)
        enc_path = os.path.join(model_path, 'label_encoder.npz')
        np.savez(enc_path, classes=label_encoder.classes_)

    return label_encoder, num_class


def load_label_encoder(encoder_path):
    """Load label encoder from saved npz."""
    data = np.load(encoder_path, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = data['classes']
    return le


def get_target_class_from_test_data(
    test_data_path,
    label_encoder_path,
    drug_key='perturbation',
    dose_key='dose_value',
):
    """
    Get the target class id for classifier guidance from test set IFN cells.
    Returns the dominant drug+dose class; if unseen, returns first non-Control class.
    """
    adata = sc.read_h5ad(test_data_path)
    label_encoder = load_label_encoder(label_encoder_path)
    classes_list = list(label_encoder.classes_)

    ifn_mask = adata.obs['perturbation_status'] == 'IFN'
    if not ifn_mask.any():
        # No IFN cells, use first non-Control
        for i, c in enumerate(classes_list):
            if c != 'Control':
                return i
        return 0

    ifn_labels = _build_drug_dose_labels(
        adata[ifn_mask], drug_key, dose_key
    )
    from collections import Counter
    counts = Counter(ifn_labels)
    dominant = counts.most_common(1)[0][0]

    if dominant in classes_list:
        return list(classes_list).index(dominant)
    for i, c in enumerate(classes_list):
        if c != 'Control':
            return i  # fallback to first IFN class
    return 0


def stabilize(expression_matrix):
    ''' Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    '''
    from scipy import optimize
    phi_hat, _ = optimize.curve_fit(lambda mu, phi: mu + phi * mu ** 2, expression_matrix.mean(1), expression_matrix.var(1))

    return np.log(expression_matrix + 1. / (2 * phi_hat[0]))

def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder

def load_data(
    *,
    data_dir,
    batch_size,
    vae_path = None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
    use_drug_cond=False,
    drug_key='perturbation',
    dose_key='dose_value',
    label_encoder=None,
    label_encoder_path=None,
    label_key='perturbation_status',
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    start_time = time.time()
    adata = sc.read_h5ad(data_dir)

    adata.var_names_make_unique()

    def _get_raw_labels():
        if use_drug_cond and drug_key in adata.obs and dose_key in adata.obs:
            return _build_drug_dose_labels(adata, drug_key, dose_key)
        if label_key and label_key in adata.obs:
            return np.array([str(x).strip() for x in adata.obs[label_key].values])
        return adata.obs['perturbation_status'].values

    if label_encoder is not None:
        raw_labels = _get_raw_labels()
        classes = np.array([
            list(label_encoder.classes_).index(l) if l in label_encoder.classes_ else 0
            for l in raw_labels
        ], dtype=np.int64)
    elif use_drug_cond and drug_key in adata.obs and dose_key in adata.obs:
        raw_labels = _build_drug_dose_labels(adata, drug_key, dose_key)
        label_encoder = LabelEncoder()
        label_encoder.fit(raw_labels)
        classes = label_encoder.transform(raw_labels)
    elif label_encoder_path is not None and os.path.exists(label_encoder_path):
        label_encoder = load_label_encoder(label_encoder_path)
        raw_labels = _get_raw_labels()
        classes = np.array([
            list(label_encoder.classes_).index(l) if l in label_encoder.classes_ else 0
            for l in raw_labels
        ], dtype=np.int64)
    else:
        raw_labels = _get_raw_labels()
        label_encoder = LabelEncoder()
        label_encoder.fit(raw_labels)
        classes = label_encoder.transform(raw_labels)

    start_time = time.time()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if scipy.sparse.issparse(adata.X):
        start_time = time.time()
        cell_data = adata.X.toarray()
    else:
        cell_data = adata.X

    if not train_vae:
        num_gene = cell_data.shape[1]
        autoencoder = load_VAE(vae_path,num_gene,hidden_dim)
        cell_data = autoencoder(torch.tensor(cell_data).float().cuda(), return_latent=True)
        cell_data = cell_data.cpu().detach().numpy()

    dataset = CellDataset(
        cell_data,
        classes
    )

    start_time = time.time()
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )

    while True:
        yield from loader

class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict

if __name__ == "__main__":
    import scanpy as sc
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import scipy.sparse

    data_path = "../../../data/fig1/task2/task2_train_random1_bulkRNAseq_exp.h5ad"

    print("[Debug] Reading h5ad...")
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()

    classes = adata.obs['perturbation_status'].values
    le = LabelEncoder()
    classes = le.fit_transform(classes)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if scipy.sparse.issparse(adata.X):
        cell_data = adata.X.toarray()
    else:
        cell_data = adata.X

    print(f"[Debug] cell_data shape: {cell_data.shape}")
    print(f"[Debug] classes shape: {classes.shape}")

    ds = CellDataset(cell_data, classes)
    print("__len__ =", len(ds))
    print("__getitem__(0) =", ds[0])