import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
import scipy
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder
import time

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
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    start_time = time.time()
    adata = sc.read_h5ad(data_dir)

    adata.var_names_make_unique()

    classes = adata.obs['perturbation_status'].values
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)

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