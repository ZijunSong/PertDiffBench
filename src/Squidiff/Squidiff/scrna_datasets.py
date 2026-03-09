import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def split_smiles_advanced(smiles):
    """
    智能拆分 SMILES：按顶层 ; 分隔，但保留 [] 内的内容
    """
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
            # 只有在方括号外，分号才是分隔符
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
    adopted from PRnet @Author: Xiaoning Qi.
    Encode SMILES of drug to rFCFP fingerprint (with safety checks)
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits), dtype=np.float32)

    for i, smiles in enumerate(drug_SMILES_list):
        # 跳过空值
        if not smiles or smiles == '' or pd.isna(smiles):
            continue
        
        # 使用智能拆分，正确处理 [Na+] 等离子
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
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                combined_fingerprint += fcfp4_list
                valid_parts += 1
                
            except Exception as e:
                print(f"Warning: Error processing SMILES part '{smi}': {e}")
                continue
        
        # 应用剂量缩放
        if valid_parts > 0:
            try:
                dose_val = float(dose_list[i]) if dose_list[i] not in ['', None] else 0.0
                if dose_val > 0:
                    combined_fingerprint = combined_fingerprint * np.log10(dose_val + 1)
                fcfp4_array[i] = combined_fingerprint
            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid dose value at index {i}: {dose_list[i]}")
                fcfp4_array[i] = combined_fingerprint
    
    return fcfp4_array

class AnnDataDataset(Dataset):
    def __init__(self, adata, control_adata=None,use_drug_structure=False,comb_num=1):
        self.use_drug_structure = use_drug_structure
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        
        if self.use_drug_structure:
            if type(control_adata.X)==np.ndarray:
                self.control_features = torch.tensor(control_adata.X, dtype=torch.float32)
            else:
                self.control_features = torch.tensor(control_adata.X.toarray(), dtype=torch.float32)
            
            smiles_series = adata.obs['smiles'].astype(str).replace({
                'nan': '', 'NaN': '', 'None': '', 'null': ''
            })
            dose_series = adata.obs['dose_value'].astype(str).replace({
                'nan': '0', 'NaN': '0', 'None': '0', 'null': '0'
            })
            
            self.drug_type_list = smiles_series.to_list()
            self.dose_list = [float(x) if x != '' else 0.0 for x in dose_series.to_list()]

            self.encoded_obs_tensor = adata.obs['perturbation_status'].copy().values
            
            self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list, comb_num=comb_num)
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        else:
            self.encoded_obs_tensor = adata.obs['perturbation_status'].copy().values
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.use_drug_structure:
            feature = self.features[idx]
            drug_dose = self.encode_drug_doses[idx]
            group = self.encoded_obs_tensor[idx]
            
            # 如果当前细胞是 control，直接使用自身特征
            # 如果当前细胞是 treated，随机采样一个 control 细胞（或取平均）
            if group == 0:  # control
                control_feature = feature  # 或者从 control_features 中找对应
            else:  # treated
                # 随机采样一个 control 细胞
                control_idx = torch.randint(0, len(self.control_features), (1,)).item()
                control_feature = self.control_features[control_idx]
            
            return {
                'feature': feature,
                'drug_dose': drug_dose,
                'group': group,
                'control_feature': control_feature
            }
        else:
            return {'feature': self.features[idx], 'group': self.encoded_obs_tensor[idx]}
            
    

def prepared_data(data_dir=None,control_data_dir=None, batch_size=64,use_drug_structure=False,comb_num=1):
     
    
    train_adata = sc.read_h5ad(data_dir)
    if use_drug_structure:
        control_adata = sc.read_h5ad(control_data_dir)
    else:
        control_adata = None
    
    _data_dataset = AnnDataDataset(train_adata,control_adata,use_drug_structure,comb_num)


    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
        
    return dataloader