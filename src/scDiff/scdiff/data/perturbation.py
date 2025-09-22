from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import os.path as osp

import torch
from sklearn.preprocessing import LabelEncoder

from scdiff.data.base import FullDatasetMixin, TargetDataset

URL_DICT = {
    "train_pbmc": "https://www.dropbox.com/s/wk5zewf2g1oat69/train_pbmc.h5ad?dl=1",
    "valid_pbmc": "https://www.dropbox.com/s/nqi971n0tk4nbfj/valid_pbmc.h5ad?dl=1",

    "train_hpoly": "https://www.dropbox.com/s/7ngt0hv21hl2exn/train_hpoly.h5ad?dl=1",
    "valid_hpoly": "https://www.dropbox.com/s/bp6geyvoz77hpnz/valid_hpoly.h5ad?dl=1",

    "train_salmonella": "https://www.dropbox.com/s/9ozdwdi37wrz9r1/train_salmonella.h5ad?dl=1",
    "valid_salmonella": "https://www.dropbox.com/s/z5jnq4nthierdgq/valid_salmonella.h5ad?dl=1",
}

RAW_FILE_NAME_DICT = {
    "pbmc": "GSE96583.h5ad", # HelenaLC/muscData/Kang18_8vs8, zellkonverter.SCE2AnnData
    "hpoly": "GSE92332_SalmHelm_UMIcounts.txt.gz",
    "salmonella": "GSE92332_SalmHelm_UMIcounts.txt.gz",
    "fig1_task1_B": "task1_train_B_exp.h5ad",
    "fig1_task1_CD4T": "task1_train_CD4T_exp.h5ad",
    "fig1_task1_CD8T": "task1_train_CD8T_exp.h5ad",
    "fig1_task1_CD14+Mono": "task1_train_CD14+Mono_exp.h5ad",
    "fig1_task1_Dendritic": "task1_train_Dendritic_exp.h5ad",
    "fig1_task1_FCGR3A+Mono": "task1_train_FCGR3A+Mono_exp.h5ad",
    "fig1_task1_NK": "task1_train_NK_exp.h5ad",
    "fig1_task2_random1": "task2_train_random1_bulkRNAseq_exp.h5ad",
    "fig1_task2_random2": "task2_train_random2_bulkRNAseq_exp.h5ad",
    "fig1_task2_random3": "task2_train_random3_bulkRNAseq_exp.h5ad",
    "fig1_task1_CD4T_6998": "CD4T_train_HVG_6998.h5ad",
    "fig1_task1_CD4T_6000": "CD4T_train_HVG_6000.h5ad",
    "fig1_task1_CD4T_5000": "CD4T_train_HVG_5000.h5ad",
    "fig1_task1_CD4T_4000": "CD4T_train_HVG_4000.h5ad",
    "fig1_task1_CD4T_3000": "CD4T_valid_HVG_3000.h5ad",
    "fig1_task1_CD4T_2000": "CD4T_valid_HVG_2000.h5ad",
    "fig1_task1_CD4T_1000": "CD4T_valid_HVG_1000.h5ad",
    "fig1_task3_mix2": "task3_test_mix2_exp.h5ad",
    "fig1_task3_mix3": "task3_test_mix3_exp.h5ad",
    "fig1_task3_mix4": "task3_test_mix4_exp.h5ad",
    "fig1_task3_mix5": "task3_test_mix5_exp.h5ad",
    "fig1_task3_mix6": "task3_test_mix6_exp.h5ad",
    "fig1_task3_mix7": "task3_test_mix7_exp.h5ad",
    "ACTA2_control_coculture": "task4_ACTA2_control_coculture_train.h5ad",
    "ACTA2_control_ifn": "task4_ACTA2_control_ifn_train.h5ad",
    "B2M_control_coculture": "task4_B2M_control_coculture_train.h5ad",
    "B2M_control_ifn": "task4_B2M_control_ifn_train.h5ad",
    "task4_ACTA2_control_to_ifn": "task4_ACTA2_control_to_ifn.h5ad",
    "task4_B2M_control_to_ifn": "task4_B2M_control_to_ifn.h5ad",
    "seed123": "seed123_control_train.h5ad",
    "seed345": "seed345_control_train.h5ad",
    "seed567": "seed567_control_train.h5ad",
    "pig_control_ifn": "pig_control_ifn.h5ad",
    "rabbit_control_ifn": "rabbit_control_ifn.h5ad",
    "rat_control_ifn": "rat_control_ifn.h5ad",
    "fig1_task1_CD4T_noise_0.1": "task1_train_CD4T_exp_noise_std_0.1.h5ad",
    "fig1_task1_CD4T_noise_0.5": "task1_train_CD4T_exp_noise_std_0.5.h5ad",
    "fig1_task1_CD4T_noise_0.25": "task1_train_CD4T_exp_noise_std_0.25.h5ad",
    "fig1_task1_CD4T_noise_1.0": "task1_train_CD4T_exp_noise_std_1.0.h5ad",
    "fig1_task1_CD4T_noise_1.5": "task1_train_CD4T_exp_noise_std_1.5.h5ad",
    "train_fig1_task1_CD4T": "task1_train_CD4T_exp.h5ad",
    "train_fig1_task1_CD4T_6000": "CD4T_train_HVG_6000.h5ad",
    "train_fig1_task1_CD4T_5000": "CD4T_train_HVG_5000.h5ad",
    "train_fig1_task1_CD4T_4000": "CD4T_train_HVG_4000.h5ad",
    "train_fig1_task1_CD4T_3000": "CD4T_train_HVG_3000.h5ad",
    "train_fig1_task1_CD4T_2000": "CD4T_train_HVG_2000.h5ad",
    "train_fig1_task1_CD4T_1000": "CD4T_train_HVG_1000.h5ad",
}

DEFAULT_CELL_TYPE_DICT = {
    "pbmc": ["CD4 T cells"], # ["CD4T"],
    "hpoly": ["TA.Early"],
    "salmonella": ["TA.Early"],
    "fig1_task1_B": ["B"],
    "fig1_task1_CD4T": ["CD4T"],
    "fig1_task1_CD8T": ["CD8T"],
    "fig1_task1_CD14+Mono": ["CD14+Mono"],
    "fig1_task1_Dendritic": ["Dendritic"],
    "fig1_task1_FCGR3A+Mono": ["FCGR3A+Mono"],
    "fig1_task2_random1": ["bulkRNAseq"],
    "fig1_task2_random2": ["bulkRNAseq"],
    "fig1_task2_random3": ["bulkRNAseq"],
    "fig1_task1_NK": ["NK"],
    "fig1_task1_CD4T_6998": ["CD4T"],
    "fig1_task1_CD4T_6000": ["CD4T"],
    "fig1_task1_CD4T_5000": ["CD4T"],
    "fig1_task1_CD4T_4000": ["CD4T"],
    "fig1_task1_CD4T_3000": ["CD4T"],
    "fig1_task1_CD4T_2000": ["CD4T"],
    "fig1_task1_CD4T_1000": ["CD4T"],
    "fig1_task3_mix2": ["mix2"],
    "fig1_task3_mix3": ["mix3"],
    "fig1_task3_mix4": ["mix4"],
    "fig1_task3_mix5": ["mix5"],
    "fig1_task3_mix6": ["mix6"],
    "fig1_task3_mix7": ["mix7"],
    "ACTA2_control_coculture": ["melanocytes"],
    "ACTA2_control_ifn": ["melanocytes"],
    "B2M_control_coculture": ["melanocytes"],
    "B2M_control_ifn": ["melanocytes"],
    "task4_ACTA2_control_to_ifn": ["melanocytes"],
    "task4_B2M_control_to_ifn": ["melanocytes"],
    "seed123": ["mammary epithelial cells"],
    "seed345": ["mammary epithelial cells"],
    "seed567": ["mammary epithelial cells"],
    "pig_control_ifn": ["species"],
    "rabbit_control_ifn": ["species"],
    "rat_control_ifn": ["species"],
    "fig1_task1_CD4T_noise_0.1": ["CD4T"],
    "fig1_task1_CD4T_noise_0.5": ["CD4T"],
    "fig1_task1_CD4T_noise_0.25": ["CD4T"],
    "fig1_task1_CD4T_noise_1.0": ["CD4T"],
    "fig1_task1_CD4T_noise_1.5": ["CD4T"],
    "train_fig1_task1_CD4T_6998": ["CD4T"],
    "train_fig1_task1_CD4T_6000": ["CD4T"],
    "train_fig1_task1_CD4T_5000": ["CD4T"],
    "train_fig1_task1_CD4T_4000": ["CD4T"],
    "train_fig1_task1_CD4T_3000": ["CD4T"],
    "train_fig1_task1_CD4T_2000": ["CD4T"],
    "train_fig1_task1_CD4T_1000": ["CD4T"],
}

PERT_DICT = {
    "pbmc": ("ctrl", "stim"), # ("control", "stimulated"),
    "hpoly": ("Control", "Hpoly.Day10"),
    "salmonella": ("Control", "Salmonella"),
    "fig1_task1_B": ("Control", "IFN"),
    "fig1_task1_CD4T": ("Control", "IFN"),
    "fig1_task1_CD8T": ("Control", "IFN"),
    "fig1_task1_CD14+Mono": ("Control", "IFN"),
    "fig1_task1_Dendritic": ("Control", "IFN"),
    "fig1_task1_FCGR3A+Mono": ("Control", "IFN"),
    "fig1_task1_NK": ("Control", "IFN"),
    "fig1_task2_random1": ("Control", "IFN"),
    "fig1_task2_random2": ("Control", "IFN"),
    "fig1_task2_random3": ("Control", "IFN"),
    "fig1_task1_CD4T_6998": ("Control", "IFN"),
    "fig1_task1_CD4T_6000": ("Control", "IFN"),
    "fig1_task1_CD4T_5000": ("Control", "IFN"),
    "fig1_task1_CD4T_4000": ("Control", "IFN"),
    "fig1_task1_CD4T_3000": ("Control", "IFN"),
    "fig1_task1_CD4T_2000": ("Control", "IFN"),
    "fig1_task1_CD4T_1000": ("Control", "IFN"),
    "fig1_task3_mix2": ("Control", "IFN"),
    "fig1_task3_mix3": ("Control", "IFN"),
    "fig1_task3_mix4": ("Control", "IFN"),
    "fig1_task3_mix5": ("Control", "IFN"),
    "fig1_task3_mix6": ("Control", "IFN"),
    "fig1_task3_mix7": ("Control", "IFN"),
    "ACTA2_control_coculture": ("Control", "IFN"),
    "ACTA2_control_ifn": ("Control", "IFN"),
    "B2M_control_coculture": ("Control", "IFN"),
    "B2M_control_ifn": ("Control", "IFN"),
    "task4_ACTA2_control_to_ifn": ("Control", "IFN"),
    "task4_B2M_control_to_ifn": ("Control", "IFN"),
    "seed123": ("Control", "IFN"),
    "seed345": ("Control", "IFN"),
    "seed567": ("Control", "IFN"),
    "pig_control_ifn": ("Control", "IFN"),
    "rabbit_control_ifn": ("Control", "IFN"),
    "rat_control_ifn": ("Control", "IFN"),
    "fig1_task1_CD4T_noise_0.1": ("Control", "IFN"),
    "fig1_task1_CD4T_noise_0.5": ("Control", "IFN"),
    "fig1_task1_CD4T_noise_0.25": ("Control", "IFN"),
    "fig1_task1_CD4T_noise_1.0": ("Control", "IFN"),
    "fig1_task1_CD4T_noise_1.5": ("Control", "IFN"),
    "train_fig1_task1_CD4T_6998": ("Control", "IFN"),
    "train_fig1_task1_CD4T_6000": ("Control", "IFN"),
    "train_fig1_task1_CD4T_5000": ("Control", "IFN"),
    "train_fig1_task1_CD4T_4000": ("Control", "IFN"),
    "train_fig1_task1_CD4T_3000": ("Control", "IFN"),
    "train_fig1_task1_CD4T_2000": ("Control", "IFN"),
    "train_fig1_task1_CD4T_1000": ("Control", "IFN"),
}


def download_data(datadir='./data', dataset='pbmc'):
    train_path = osp.join(datadir, f"train_{dataset}.h5ad")
    valid_path = osp.join(datadir, f"valid_{dataset}.h5ad")

    train_url = URL_DICT[f"train_{dataset}"]
    valid_url = URL_DICT[f"valid_{dataset}"]

    import wget
    if not osp.exists(train_path):
        print(train_url)
        print(train_path)
        wget.download(train_url, train_path)
    if not osp.exists(valid_path):
        print(valid_url)
        print(valid_path)
        wget.download(valid_url, valid_path)

    print(f"{dataset} data has been downloaded and saved in {datadir}")


class PerturbationBase(ABC):
    def __init__(self, datadir='data/fig2/task3_cross_species', dataset='pbmc', fname='task1_train_B_exp.h5ad', test_cell_types=None, 
                 save_processed=True, splits={'train':0.9, 'valid':0.1}, post_cond_flag=True, force_split=False,
                 ignore_cond_flag=False, normalize=True, return_raw=False, highly_variable=False, seed=10,
                 pretrained_gene_list=None, pretrained_gene_list_path=None, subset_flag=False):
        self.celltype_key = 'Cell.Type'
        self.batch_key = 'batch'
        self.pert_key = 'perturbation_status'
        self.ctrl_key, self.stim_key = PERT_DICT[dataset]
        self.datadir = datadir
        self.dataset = dataset
        self.normalize = normalize
        self.return_raw = return_raw
        self.subset_flag = subset_flag
        self.save_processed = save_processed
        self.post_cond_flag = post_cond_flag
        self.highly_variable = highly_variable
        self.test_cell_types = test_cell_types
        self.ignore_cond_flag = ignore_cond_flag
        if pretrained_gene_list is None and pretrained_gene_list_path is not None:
            assert pretrained_gene_list_path.endswith('npy')
            pretrained_gene_list = np.load(pretrained_gene_list_path, allow_pickle=True)
        self.pretrained_gene_list = pretrained_gene_list
        self._read(datadir=datadir, dataset=dataset, fname=fname, normalize=normalize)
        self._prepare_split(splits=splits, seed=seed, fname=fname, force_split=force_split)
        self._init_condiitons()
        self._prepare()

    def _read(self, datadir='data/scrna_data', dataset='pbmc', fname='Perturbation_processed.h5ad', normalize=True):
        print("######")
        print(osp.join(datadir, fname))
        if osp.exists(osp.join(datadir, fname)) and fname.endswith('.h5ad'):
            self.adata = ad.read_h5ad(osp.join(datadir, fname))
        else:
            download_data(datadir, dataset)
            adata_train = ad.read_h5ad(osp.join(datadir, f"train_{dataset}.h5ad"))
            adata_valid = ad.read_h5ad(osp.join(datadir, f"valid_{dataset}.h5ad"))
            if dataset == 'salmonella': # # salmonella requires anndata < 0.8
                adata_train.obs.index = adata_train.obs.index.astype(str)
                adata_valid.obs.index = adata_valid.obs.index.astype(str)
                adata_train.var.index = adata_train.var.index.astype(str)
                adata_valid.var.index = adata_valid.var.index.astype(str)
                for col in adata_train.obs.columns:
                    adata_train.obs[col] = adata_train.obs[col].astype(str)
                    adata_valid.obs[col] = adata_valid.obs[col].astype(str)

            if dataset == 'pbmc':
                self.adata = ad.read_h5ad(osp.join(datadir, RAW_FILE_NAME_DICT[dataset]))
                self.adata.obs[self.celltype_key] = self.adata.obs['cell'].copy()
                self.adata.obs[self.batch_key] = self.adata.obs['multiplets'].copy()
                self.adata.obs[self.pert_key] = self.adata.obs['stim'].copy()
                sc.pp.filter_genes(self.adata, min_cells=5)
                sc.pp.filter_cells(self.adata, min_genes=500)

            else:
                # adata_train.obs['train_valid_split'] = 'train'
                # adata_valid.obs['train_valid_split'] = 'valid'
                self.adata = adata_train.concatenate(adata_valid) # for anndata < 0.8
                self.adata.obs_names_make_unique()

                raw_df = pd.read_csv(
                    osp.join(
                        datadir, 
                        RAW_FILE_NAME_DICT[dataset]  # "salmonella": "GSE92332_SalmHelm_UMIcounts.txt.gz"
                    ),
                sep='\t').T
                obs_index = self.adata.obs.index
                obs_index = [x.split('-')[0] for x in obs_index]
                self.adata.obs.index = obs_index
                raw = ad.AnnData(raw_df, dtype=np.float32)[obs_index]
                raw.obs = self.adata.obs
                self.adata = raw

            self.adata.var['highly_variable'] = [x in adata_train.var.index for x in self.adata.var.index]
            self.adata.layers['counts'] = self.adata.X.copy()
            if normalize:
                sc.pp.normalize_total(self.adata, target_sum=1e4, key_added='library_size')
                sc.pp.log1p(self.adata)
        
        if self.pretrained_gene_list is not None:
            self.gene_list = self.adata.var.index.to_list()
            self.gene_list = [x for x in self.gene_list if x in self.pretrained_gene_list]
            if self.subset_flag:
                self.adata = self.adata[:, self.gene_list]

    def _prepare_split(self, splits={'train':0.9, 'valid':0.1}, force_split=False, seed=10, 
                       fname='Perturbation_processed.h5ad'):
        if not (
            'train_valid_split' in self.adata.obs.columns 
            and sorted(splits) == sorted(np.unique(self.adata.obs['train_valid_split'])) 
            and not force_split
        ):
            rng = np.random.default_rng(seed)
            N = len(self.adata)
            perm = rng.permutation(range(N))
            self.adata.obs['train_valid_split'] = 'train'
            self.adata.obs['train_valid_split'][
                perm[int(N * splits['train']):int(N * (splits['train'] + splits['valid']))]
            ] = 'valid'

        if self.test_cell_types is None:
            self.test_cell_types = DEFAULT_CELL_TYPE_DICT[self.dataset]
        print("######")
        print(self.celltype_key)  # cell_label
        self.adata.obs[self.celltype_key] = self.adata.obs[self.celltype_key].astype(str)

        assert all([x in np.unique(self.adata.obs[self.celltype_key]) for x in self.test_cell_types])

        self.adata.obs['split'] = self.adata.obs['train_valid_split'].astype(str)
        print("### Keys:")
        print(self.pert_key)
        print(self.stim_key)
        self.adata.obs['split'][
            (self.adata.obs[self.celltype_key].isin(self.test_cell_types)) &
            (self.adata.obs[self.pert_key] == self.stim_key)
        ] = 'test'
        if self.save_processed and fname is not None:
            import scipy
            if not scipy.sparse.issparse(self.adata.X):
                self.adata.X = scipy.sparse.csr_matrix(self.adata.X)
            if not scipy.sparse.issparse(self.adata.layers['counts']):
                self.adata.layers['counts'] = scipy.sparse.csr_matrix(self.adata.layers['counts'])
            print(f"Saving processed file to {osp.join(self.datadir, fname)}")
            self.adata.write_h5ad(osp.join(self.datadir, fname), compression='gzip')

    def _init_condiitons(self):
        self.celltype_enc = LabelEncoder()
        self.celltype_enc.classes_ = np.array(
            ["null"] + sorted(self.adata.obs[self.celltype_key].astype(str).unique())
        )

        if self.batch_key not in self.adata.obs.columns:
            print(f"[Info] batch_key '{self.batch_key}' not found in data. Using single batch 'null'.")
            self.adata.obs[self.batch_key] = "null"

        self.batch_enc = LabelEncoder()
        self.batch_enc.fit(self.adata.obs[self.batch_key].astype(str))

        self.pert_enc = LabelEncoder()
        self.pert_enc.classes_ = np.array([self.ctrl_key, self.stim_key])

        if self.post_cond_flag:
            self.cond_num_dict = {
                'cell_type': len(self.celltype_enc.classes_),
                'pert': len(self.pert_enc.classes_),
            }
            self.post_cond_num_dict = {'batch': len(self.batch_enc.classes_)}
        else:
            self.cond_num_dict = {
                'batch': len(self.batch_enc.classes_),
                'cell_type': len(self.celltype_enc.classes_),
                'pert': len(self.pert_enc.classes_),
            }
            self.post_cond_num_dict = None

    def _load(self):
        if self.highly_variable:
            self.adata = self.adata[:, self.adata.var.highly_variable]
        print(self.SPLIT)
        if self.SPLIT == 'test':
            adata_input = self.adata[
                (self.adata.obs[self.celltype_key].isin(self.test_cell_types)) &
                (self.adata.obs[self.pert_key] == self.ctrl_key)
            ]
            adata_target = self.adata[self.adata.obs["split"] == self.SPLIT]
            # self.input = torch.tensor(adata_input.X.A).float()
            # self.target = torch.tensor(adata_target.X.A).float()
            self.input = torch.tensor(adata_input.X.toarray()).float()
            self.target = torch.tensor(adata_target.X.toarray()).float()
            self.adata = adata_input.copy()
            self.adata.obs[self.pert_key] = self.stim_key
        else:  ## train
            # self.input = torch.tensor(self.adata.X.A).float()
            self.input = torch.tensor(self.adata.X.toarray()).float()
            self.target = None

        self.gene_names = self.adata.var.index.tolist()
        self.celltype = self.celltype_enc.transform(self.adata.obs[self.celltype_key].astype(str))
        self.batch = self.batch_enc.transform(self.adata.obs[self.batch_key].astype(str))
        print("### self.pert_key")
        print(self.pert_key) # perturbation_status
        self.pert = self.pert_enc.transform(self.adata.obs[self.pert_key].astype(str))
        self.cond = {
            'batch': torch.tensor(self.batch).float(),
            'cell_type': torch.tensor(self.celltype).float(),
            'pert': torch.tensor(self.pert).float(),
        }

        if self.pretrained_gene_list is not None:
            pretrained_gene_index = dict(zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list)))))
            self.input_gene_idx = torch.tensor([
                pretrained_gene_index[o] for o in self.gene_list
                if o in pretrained_gene_index
            ]).long()

    @abstractmethod
    def _prepare(self):
        ...


class PerturbationTrain(TargetDataset, PerturbationBase):
    SPLIT = "train"
    TARGET_KEY = "pert_target"


class PerturbationValidation(TargetDataset, PerturbationBase):
    SPLIT = "valid"
    TARGET_KEY = "pert_target"


class PerturbationTest(TargetDataset, PerturbationBase):
    SPLIT = "test"
    TARGET_KEY = "pert_target"

    def _prepare(self):
        self._load()


class Perturbation(FullDatasetMixin, TargetDataset, PerturbationBase):
    ...
