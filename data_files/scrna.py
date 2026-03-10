# data/scrna.py
import numpy as np
import torch
from torch.utils.data import Dataset
import anndata
import pandas as pd # 确保导入 pandas
from sklearn.preprocessing import LabelEncoder


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


class PairedScrnaDataset(Dataset):
    """
    一个智能的数据集类，用于创建（对照组 vs 处理组）的细胞对。
    它会自动检测数据中是否存在捐赠者/批次信息，并采取相应的配对策略。
    支持 scGen setting：通过 pair_only_obs_key / pair_only_obs_value 仅在部分细胞（如 train）上做一一配对，
    其余细胞（如 test_control）仅参与数据体量，映射为分布到分布。
    """
    def __init__(self, adata_path, donor_key=None, ctrl_status='Control', pert_status='IFN',
                 pair_only_obs_key=None, pair_only_obs_value=None):
        """
        Args:
            adata_path (str): .h5ad 文件的路径。
            donor_key (str, optional): 手动指定用于分组的列名（例如 'donor', 'batch'）。
            ctrl_status (str, optional): 表示对照组在'perturbation_status'列中的值。
            pert_status (str, optional): 表示处理组在'perturbation_status'列中的值。
            pair_only_obs_key (str, optional): 若与 pair_only_obs_value 同时给出，则仅在
                obs[pair_only_obs_key] == pair_only_obs_value 的细胞中构建配对（用于 scGen：仅 train 配对）。
            pair_only_obs_value (str, optional): 见 pair_only_obs_key。
        """
        adata = anndata.read_h5ad(adata_path)
        obs = adata.obs.copy()
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        # scGen setting：只在与 pair_only 匹配的子集上构建配对
        if pair_only_obs_key is not None and pair_only_obs_value is not None and pair_only_obs_key in obs.columns:
            obs_for_pair = obs[obs[pair_only_obs_key].astype(str) == str(pair_only_obs_value)]
            print(f"INFO: scGen/pair-only 模式，仅在 obs['{pair_only_obs_key}']=='{pair_only_obs_value}' 的细胞上配对 (n={len(obs_for_pair)})。")
        else:
            obs_for_pair = obs

        self.pairs = []
        
        # --- 智能配对逻辑开始（在 obs_for_pair 上）---
        
        donor_key_found = None
        
        if donor_key_found:
            print(f"INFO: 正在按 '{donor_key_found}' 列进行精细配对...")
            for group_id, sub_obs in obs_for_pair.groupby(donor_key_found):
                idx_ctrl = sub_obs[sub_obs['perturbation_status'] == ctrl_status].index
                idx_pert = sub_obs[sub_obs['perturbation_status'] == pert_status].index
                if len(idx_ctrl) > 0 and len(idx_pert) > 0:
                    n = min(len(idx_ctrl), len(idx_pert))
                    for i in range(n):
                        self.pairs.append((idx_ctrl[i], idx_pert[i]))
        else:
            # 策略B：全局配对（在 obs_for_pair 内）
            if pair_only_obs_key is None:
                print("INFO: 未找到可用的捐赠者/批次键。假设所有细胞来自同一组，进行全局配对。")
            idx_ctrl = obs_for_pair[obs_for_pair['perturbation_status'] == ctrl_status].index
            idx_pert = obs_for_pair[obs_for_pair['perturbation_status'] == pert_status].index
            if len(idx_ctrl) > 0 and len(idx_pert) > 0:
                n = min(len(idx_ctrl), len(idx_pert))
                for i in range(n):
                    self.pairs.append((idx_ctrl[i], idx_pert[i]))

        # --- 智能配对逻辑结束 ---

        if not self.pairs:
            print(f"\n严重警告：未能生成任何配对样本！请检查 'perturbation_status' 列中是否存在 '{ctrl_status}' 和 '{pert_status}' 的值。")
        else:
            print(f"\n成功生成 {len(self.pairs)} 对配对样本。")

        self.X = X
        # 为了后续getitem能快速查找，将obs的index设为索引
        self.obs = obs.set_index(pd.Index(obs.index))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        i0, i1 = self.pairs[i]
        # 使用 .loc 进行更可靠的索引
        v0 = self.X[self.obs.index.get_loc(i0)]
        v1 = self.X[self.obs.index.get_loc(i1)]
        return torch.from_numpy(v0).float(), torch.from_numpy(v1).float()

class PairedScrnaDatasetDrugCond(Dataset):
    """
    Paired (Control, IFN) dataset with drug+dose conditioning for MOA task.
    Returns (ctrl_expr, pert_expr, drug_label_idx, dose_scalar) per sample.
    """
    def __init__(self, adata_path, drug_key='perturbation', dose_key='dose_value',
                 ctrl_status='Control', pert_status='IFN'):
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
        print(f"PairedScrnaDatasetDrugCond: {len(self.pairs)} pairs from {adata_path}")

        self.X = X
        self.obs = obs.set_index(obs.index)
        self.drug_key = drug_key
        self.dose_key = dose_key

        labels = _build_drug_dose_labels(obs, drug_key, dose_key)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.label_indices = self.label_encoder.transform(labels)
        self.dose_values = obs[dose_key].astype(float).fillna(0).values if dose_key in obs.columns else np.zeros(len(obs))

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