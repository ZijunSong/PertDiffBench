"""
Perturbation dataset with drug and dose as additional conditions.

Supports h5ad with columns:
  - perturbation_status: control vs perturbed (same as pert_key)
  - perturbation: drug name
  - dose_value: drug dose (will be discretized for embedding)

Usage: extend your config to use scdiff.data.perturbation_drug.PerturbationDrugTrain/Test
and add use_drug_cond: true, drug_key: perturbation, dose_key: dose_value.
"""
import os.path as osp
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import anndata as ad

from scdiff.data.perturbation import (
    PerturbationBase, PerturbationTrain, PerturbationValidation, PerturbationTest,
    Perturbation, PERT_DICT, DEFAULT_CELL_TYPE_DICT, download_data,
)
from scdiff.data.base import FullDatasetMixin, TargetDataset


def _discretize_dose(dose_series, n_bins=5):
    """Discretize continuous dose into bins. Returns integer bin indices (0..n_bins-1)."""
    dose = np.asarray(dose_series, dtype=float)
    valid = ~np.isnan(dose) & np.isfinite(dose)
    if not np.any(valid):
        return np.zeros(len(dose), dtype=int)
    valid_dose = dose[valid]
    try:
        bins = np.percentile(valid_dose, np.linspace(0, 100, n_bins + 1)[1:-1])
        bins = np.unique(bins)
        if len(bins) < 2:
            bins = np.array([np.min(valid_dose), np.max(valid_dose)])
        indices = np.digitize(dose, bins)
        indices = np.clip(indices, 0, len(bins))
    except Exception:
        indices = np.zeros(len(dose), dtype=int)
    return indices


def _safe_label_transform(enc, values, default_idx=0):
    """Transform values with LabelEncoder, using default_idx for unseen labels."""
    values = np.asarray(values)
    result = np.full(len(values), default_idx, dtype=np.int64)
    for i, v in enumerate(values):
        v_str = str(v)
        if v_str in enc.classes_:
            result[i] = np.where(enc.classes_ == v_str)[0][0]
    return result


class PerturbationDrugBase(PerturbationBase):
    """
    Perturbation dataset with optional drug name and dose as conditions.

    Supports custom pert_key (e.g. perturbation_status), ctrl_key, stim_key for
    datasets that use different column names/values.
    """

    def __init__(self, datadir='./data', dataset='pbmc', fname='pbmc_processed.h5ad',
                 test_cell_types=None, save_processed=True, splits={'train': 0.9, 'valid': 0.1},
                 post_cond_flag=True, force_split=False, ignore_cond_flag=False, normalize=True,
                 return_raw=False, highly_variable=True, seed=0, pretrained_gene_list=None,
                 pretrained_gene_list_path=None, subset_flag=False,
                 use_drug_cond=False, drug_key='perturbation', dose_key='dose_value',
                 dose_n_bins=5, ctrl_drug_value='control', ctrl_dose_value=0.0,
                 pert_key=None, ctrl_key=None, stim_key=None, allow_custom_dataset=False,
                 celltype_key=None):
        """
        Args:
            use_drug_cond: If True, add drug and dose as conditions.
            drug_key: obs column name for drug/perturbation name.
            dose_key: obs column name for dose value.
            dose_n_bins: Number of bins for discretizing dose.
            ctrl_drug_value: Value used for control cells in drug_key (e.g. 'control', 'DMSO').
            ctrl_dose_value: Value used for control cells in dose_key.
            pert_key: Override column for control/perturbed status (e.g. 'perturbation_status').
            ctrl_key, stim_key: Override control/perturbed values (e.g. 'control', 'treated').
            allow_custom_dataset: If True, allow dataset names outside pbmc/hpoly/salmonella.
        celltype_key: Override obs column for cell type (e.g. 'celltype'). If None and
            allow_custom_dataset, will use 'Cell.Type' if present else 'celltype'.
        """
        self.use_drug_cond = use_drug_cond
        self.drug_key = drug_key
        self.dose_key = dose_key
        self.dose_n_bins = dose_n_bins
        self.ctrl_drug_value = ctrl_drug_value
        self.ctrl_dose_value = ctrl_dose_value
        self._custom_pert_key = pert_key
        self._custom_ctrl_key = ctrl_key
        self._custom_stim_key = stim_key
        self._allow_custom_dataset = allow_custom_dataset
        if allow_custom_dataset:
            dataset = dataset if dataset in ['pbmc', 'hpoly', 'salmonella'] else 'pbmc'
            # Auto-detect celltype_key and test_cell_types from data when not provided
            if celltype_key is None or test_cell_types is None:
                fpath = osp.join(datadir, fname)
                if osp.exists(fpath) and fname.endswith('.h5ad'):
                    adata_peek = ad.read_h5ad(fpath)
                    if celltype_key is None:
                        celltype_key = 'Cell.Type' if 'Cell.Type' in adata_peek.obs.columns else 'celltype'
                    if test_cell_types is None:
                        test_cell_types = list(adata_peek.obs[celltype_key].astype(str).unique())
        super().__init__(
            datadir=datadir, dataset=dataset, fname=fname, test_cell_types=test_cell_types,
            save_processed=save_processed, splits=splits, post_cond_flag=post_cond_flag,
            force_split=force_split, ignore_cond_flag=ignore_cond_flag, normalize=normalize,
            return_raw=return_raw, highly_variable=highly_variable, seed=seed,
            pretrained_gene_list=pretrained_gene_list,
            pretrained_gene_list_path=pretrained_gene_list_path, subset_flag=subset_flag,
            celltype_key=celltype_key,
        )
        if self._custom_pert_key is not None:
            self.pert_key = self._custom_pert_key
        if self._custom_ctrl_key is not None:
            self.ctrl_key = self._custom_ctrl_key
        if self._custom_stim_key is not None:
            self.stim_key = self._custom_stim_key
        if self._custom_ctrl_key is not None or self._custom_stim_key is not None:
            self.pert_enc = LabelEncoder()
            self.pert_enc.classes_ = np.array([self.ctrl_key, self.stim_key])

    def _read(self, datadir='./data', dataset='pbmc', fname='Perturbation_processed.h5ad', normalize=True):
        if self._custom_pert_key is not None:
            self.pert_key = self._custom_pert_key
        if self._custom_ctrl_key is not None:
            self.ctrl_key = self._custom_ctrl_key
        if self._custom_stim_key is not None:
            self.stim_key = self._custom_stim_key
        return super()._read(datadir=datadir, dataset=dataset, fname=fname, normalize=normalize)

    def _init_condiitons(self):
        super()._init_condiitons()

        if not self.use_drug_cond:
            return

        if self.drug_key not in self.adata.obs.columns:
            raise ValueError(
                f"use_drug_cond=True but column '{self.drug_key}' not found in adata.obs. "
                f"Available: {list(self.adata.obs.columns)}"
            )
        if self.dose_key not in self.adata.obs.columns:
            raise ValueError(
                f"use_drug_cond=True but column '{self.dose_key}' not found in adata.obs."
            )

        drug_vals = self.adata.obs[self.drug_key].astype(str)
        uniq = sorted(set(drug_vals.unique()) - {str(self.ctrl_drug_value)})
        self.drug_enc = LabelEncoder()
        self.drug_enc.classes_ = np.array([str(self.ctrl_drug_value)] + uniq)

        dose_raw = self.adata.obs[self.dose_key].values
        dose_discrete = _discretize_dose(dose_raw, n_bins=self.dose_n_bins)
        self.dose_enc = LabelEncoder()
        self.dose_enc.classes_ = np.array(sorted(np.unique(dose_discrete)))

        n_drug = len(self.drug_enc.classes_)
        n_dose = len(self.dose_enc.classes_)

        if self.post_cond_flag:
            self.cond_num_dict['drug'] = n_drug
            self.cond_num_dict['dose'] = n_dose
        else:
            self.cond_num_dict['drug'] = n_drug
            self.cond_num_dict['dose'] = n_dose

    def _load(self):
        if self.highly_variable:
            self.adata = self.adata[:, self.adata.var.highly_variable]
        if self.SPLIT == 'test':
            adata_input = self.adata[
                (self.adata.obs[self.celltype_key].isin(self.test_cell_types)) &
                (self.adata.obs[self.pert_key] == self.ctrl_key)
            ]
            adata_target = self.adata[self.adata.obs["split"] == self.SPLIT]
            self.input = torch.tensor(adata_input.X.toarray()).float()
            self.target = torch.tensor(adata_target.X.toarray()).float()
            self.adata = adata_input.copy()
            self.adata.obs[self.pert_key] = self.stim_key
            if self.use_drug_cond:
                target_drug = adata_target.obs[self.drug_key].iloc[0]
                target_dose = adata_target.obs[self.dose_key].iloc[0]
                n_input = len(self.adata)
                self.adata.obs[self.drug_key] = pd.Series(
                    [target_drug] * n_input, index=self.adata.obs.index
                )
                self.adata.obs[self.dose_key] = pd.Series(
                    [target_dose] * n_input, index=self.adata.obs.index
                )
        else:
            self.input = torch.tensor(self.adata.X.toarray()).float()
            self.target = None

        self.gene_names = self.adata.var.index.tolist()
        self.celltype = self.celltype_enc.transform(
            self.adata.obs[self.celltype_key].astype(str)
        )
        self.batch = self.batch_enc.transform(
            self.adata.obs[self.batch_key].astype(str)
        )
        self.pert = self.pert_enc.transform(
            self.adata.obs[self.pert_key].astype(str)
        )
        self.cond = {
            'batch': torch.tensor(self.batch).float(),
            'cell_type': torch.tensor(self.celltype).float(),
            'pert': torch.tensor(self.pert).float(),
        }

        if self.use_drug_cond:
            drug_vals = self.adata.obs[self.drug_key].astype(str)
            drug_vals = drug_vals.replace('', str(self.ctrl_drug_value)).fillna(str(self.ctrl_drug_value))
            drug_encoded = _safe_label_transform(
                self.drug_enc, drug_vals.values, default_idx=0
            )
            dose_raw = self.adata.obs[self.dose_key].values
            dose_discrete = _discretize_dose(dose_raw, n_bins=self.dose_n_bins)
            dose_encoded = _safe_label_transform(
                self.dose_enc, dose_discrete.astype(str), default_idx=0
            )
            self.cond['drug'] = torch.tensor(drug_encoded).float()
            self.cond['dose'] = torch.tensor(dose_encoded).float()

        if self.pretrained_gene_list is not None:
            pretrained_gene_index = dict(
                zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list))))
            )
            self.input_gene_idx = torch.tensor([
                pretrained_gene_index[o] for o in self.gene_list
                if o in pretrained_gene_index
            ]).long()


class PerturbationDrugTrain(PerturbationDrugBase, PerturbationTrain):
    SPLIT = "train"
    TARGET_KEY = "pert_target"


class PerturbationDrugValidation(PerturbationDrugBase, PerturbationValidation):
    SPLIT = "valid"
    TARGET_KEY = "pert_target"


class PerturbationDrugTest(PerturbationDrugBase, PerturbationTest):
    SPLIT = "test"
    TARGET_KEY = "pert_target"

    def _prepare(self):
        self._load()


class PerturbationDrug(FullDatasetMixin, TargetDataset, PerturbationDrugBase):
    """Full dataset (train+valid+test) with drug/dose conditioning. MRO: avoid Perturbation to prevent duplicate PerturbationBase."""
    pass
