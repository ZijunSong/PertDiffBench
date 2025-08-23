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
    只保留在 stoi（词表）中出现的基因。
    - counts: 原始表达矩阵，shape [B, G]
    - gene_names: 原始基因列表，length G
    - stoi: 基因->idx 映射表

    返回:
    - new_counts: 仅含有效列的矩阵，shape [B, G']
    - new_gene_names: 仅含有效基因名的列表，length G'
    """
    # 找到在词表中的基因及其原始下标
    valid_pairs = [(i, g) for i, g in enumerate(gene_names) if g in stoi]
    if not valid_pairs:
        raise ValueError("没有任何基因在词表中，请检查输入数据与词表的匹配。")
    valid_indices, valid_genes = zip(*valid_pairs)
    valid_indices = list(valid_indices)
    valid_genes = list(valid_genes)

    # 截断 counts 矩阵，只保留有效的列
    new_counts = counts[:, valid_indices]
    return new_counts, valid_genes

def embed_cells(
    model: torch.nn.Module,
    tokenizer: scGPTTokenizer,
    x: torch.Tensor,
    gene_names: List[str],    # 字符串列表
) -> torch.Tensor:
    """
    1) 把基因名称映射成 vocab 中的整数索引数组 gene_ids_array (np.int64, 长度 G)；
    2) 调用 tokenize_and_pad_batch 完成 tokenize + padding；
    3) 返回 CLS token embedding。
    """
    print(f"[embed_cells DEBUG] 输入 x 的形状: {x.shape}")  # 输入 x 的形状: torch.Size([64, 2000])
    print(f"[embed_cells DEBUG] 输入 gene_names 的长度: {len(gene_names)}")  # 入 gene_names 的长度: 2000
    if gene_names:
        print(f"[embed_cells DEBUG] gene_names 示例 (前5个): {gene_names[:5]}")  # gene_names 示例 (前5个): ['HSPA1A', 'LYZ', 'CXCL8', 'SERPINB2', 'SOD2']

    device = model.device
    counts = x.detach().cpu().numpy()  # 将输入张量转换为NumPy数组，形状 (B, G)
    B, G = counts.shape  # 获取批次大小和基因数量

    print(f"[embed_cells DEBUG] counts 类型: {type(counts)}, dtype: {counts.dtype}")
    print(f"[embed_cells DEBUG] counts[0,:5] 示例: {counts[0, :5]}")

    # 1) 准备词表并注册特殊 token
    vocab = get_default_gene_vocab() # 获取默认基因词汇表
    stoi = vocab.get_stoi()
    print(f"[embed_cells DEBUG] 默认词汇表示例 (前5个stoi): {list(vocab.get_stoi().items())[:5]}")  # 默认词汇表示例 (前5个stoi): [('ZYG11B', 48287), ('ZXDC', 48284), ('ZXDB', 48283), ('ZWS1', 48281), ('ZWILCH', 48278)]
    print(f"[embed_cells DEBUG] 默认词汇表大小: {len(vocab)}")  # 默认词汇表大小: 48292
    
    # 确保特殊token在词汇表中
    for special_token in ("<pad>", "<cls>"):
        if special_token not in vocab.get_stoi():
            # 如果特殊token不在，则添加到词汇表中
            stoi[special_token] = max(stoi.values()) + 1 if stoi else 0 # 处理空词典的情况
            vocab = vocab.from_dict(stoi) # 更新词汇表
            print(f"[embed_cells DEBUG] 添加特殊token '{special_token}' 到词汇表")
            # 添加特殊token '<pad>' 到词汇表
            # 添加特殊token '<cls>' 到词汇表
        vocab.set_default_token(special_token)  # 设置默认token (虽然scGPT的tokenizer可能不直接使用这个)
    stoi = vocab.get_stoi()
    print(f"[embed_cells DEBUG] 更新后词汇表示例 (前5个stoi): {list(vocab.get_stoi().items())[:5]}")  # 更新后词汇表示例 (前5个stoi): [('ZYG11B', 48287), ('ZXDC', 48284), ('ZXDB', 48283), ('ZWS1', 48281), ('ZWILCH', 48278)]
    print(f"[embed_cells DEBUG] 更新后词汇表大小: {len(vocab)}")  # 更新后词汇表大小: 48294

    # 2) 过滤到交集数据
    counts, gene_names = _filter_to_vocab(counts, gene_names, stoi)
    B, G = counts.shape  # 更新 G
    print(f"[embed_cells DEBUG] 过滤后 G': {G}, 保留基因示例: {gene_names[:5]}")

    # —— 在这里对过滤后的 counts 归一化 ----
    counts = np.log1p(counts)
    min_vals = counts.min(axis=0, keepdims=True)
    max_vals = counts.max(axis=0, keepdims=True)
    counts = (counts - min_vals) / (max_vals - min_vals + 1e-6)
    print(f"[embed_cells DEBUG] 归一化后 counts: min={counts.min():.4f}, max={counts.max():.4f}")
    # ---------------------------------------------

    # 3) 构建 gene_ids_array
    gene_ids_array = np.array([stoi[g] for g in gene_names], dtype=np.int64)
    print(f"[embed_cells DEBUG] gene_ids_array 的形状: {gene_ids_array.shape}")
    print(f"[embed_cells DEBUG] gene_ids_array 的数据类型: {gene_ids_array.dtype}")
    print(f"[embed_cells DEBUG] gene_ids_array 示例 (前5个): {gene_ids_array[:5]}")

    # 断言检查基因数量是否匹配
    assert gene_ids_array.shape[0] == G, \
        f"基因数量 G ({G}) 与 gene_ids_array 长度 ({gene_ids_array.shape[0]}) 不匹配。可能是由于gene_names中的基因在词汇表中找不到。"

    # 4) 调用 tokenize_and_pad_batch
    # 这个函数是scGPT库的一部分，错误发生在它的内部或它调用的函数中
    print("[embed_cells DEBUG] 即将调用 tokenize_and_pad_batch...")
    print(f"[embed_cells DEBUG]   data (counts) 的形状: {counts.shape}, 类型: {type(counts)}")
    print(f"[embed_cells DEBUG]   gene_ids (gene_ids_array) 的形状: {gene_ids_array.shape}, 类型: {type(gene_ids_array)}, dtype: {gene_ids_array.dtype}")
    print(f"[embed_cells DEBUG]   max_len: {G}")
    print(f"[embed_cells DEBUG]   vocab 对象: {type(vocab)}")
    print(f"[embed_cells DEBUG]   pad_token: '<pad>'")
    print(f"[embed_cells DEBUG]   cls_token: '<cls>'")
    print(f"[embed_cells DEBUG]   pad_value: 0")
    print(f"[embed_cells DEBUG]   append_cls: True")
    print(f"[embed_cells DEBUG]   return_pt: True")

    try:
        batch = tokenize_and_pad_batch(
            data=counts,            # 输入表达数据，NumPy数组 [B, G]
            gene_ids=gene_ids_array,# 基因的整数索引数组，NumPy int64 数组 [G]
            max_len=G,              # 最大序列长度，这里等于基因数量
            vocab=vocab,            # 词汇表对象
            pad_token="<pad>",      # padding使用的token
            cls_token="<cls>",      # CLS token
            pad_value=0,            # padding使用的值
            append_cls=True,        # 是否在序列前添加CLS token
            return_pt=True,         # 是否返回PyTorch张量
        )
    except TypeError as e:
        print(f"[embed_cells ERROR] 调用 tokenize_and_pad_batch 时发生 TypeError: {e}")
        print(f"[embed_cells DEBUG HINT] 检查 tokenize_and_pad_batch 内部的 tokenize_batch 函数，其输入 gene_ids (即这里的 gene_ids_array) 和 idx。")
        print(f"[embed_cells DEBUG HINT] gene_ids_array (传递给tokenize_batch的gene_ids) 应该是一维 NumPy int64 数组。")
        print(f"[embed_cells DEBUG HINT] idx (在tokenize_batch内部生成或使用) 必须是整数标量或有效的整数数组索引。")
        raise e # 重新抛出异常，以便看到完整的Traceback

    # batch 应该是一个字典，包含 'genes' (token ID) 和 'values' (表达值或attention mask)
    print(f"[embed_cells DEBUG] tokenize_and_pad_batch 返回的 batch keys: {batch.keys()}")
    print(f"[embed_cells DEBUG] batch['genes'] 的形状: {batch['genes'].shape}, 类型: {batch['genes'].dtype}")
    print(f"[embed_cells DEBUG] batch['values'] 的形状: {batch['values'].shape}, 类型: {batch['values'].dtype}")

    # 5) 前向 scGPT 模型
    input_ids = batch["genes"].to(device)      # 获取基因token ID，并移动到设备
    values = batch["values"].to(device) # 获取表达值/mask，并移动到设备
    pad_id = stoi["<pad>"]

    # 统计总 NaN 数量和比例
    num_total = values.numel()
    nan_count = torch.isnan(values).sum().item()
    nan_ratio = nan_count / num_total
    print(f"[embed_cells DEBUG] values 中共有 {nan_count}/{num_total} 个 NaN（{nan_ratio:.2%}）")

    # 再按行（按样本）统计，看看哪些样本受影响最严重
    nan_per_row = torch.isnan(values).sum(dim=1)  # shape [B]
    rows_with_nan = (nan_per_row > 0).nonzero(as_tuple=False).view(-1).tolist()
    print(f"[embed_cells DEBUG] 有 NaN 的样本索引：{rows_with_nan[:10]}（共 {len(rows_with_nan)} 个）")
    print(f"[embed_cells DEBUG] 每个有 NaN 样本的 NaN 数量（前 10）：{nan_per_row[rows_with_nan[:10]].tolist()}")

    nan0 = torch.isnan(values).sum().item()
    if nan0 > 0:
        print(f"[embed_cells DEBUG] values 中留有 {nan0} 个 NaN，统一替换为 0")
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) 再把所有 pad 位置的 values 也置为 0，确保万无一失
    pad_mask = (input_ids == pad_id)
    if pad_mask.any():
        values = values.masked_fill(pad_mask, 0.0)

    assert not torch.isnan(values).any(), "values 中出现 NaN"
    assert not torch.isnan(input_ids.float()).any(), "input_ids 转 float 后出现 NaN"

    attention_mask = (input_ids != pad_id)

    print(f"[embed_cells DEBUG] scGPT模型输入 input_ids 的形状: {input_ids.shape}")
    print(f"[embed_cells DEBUG] scGPT模型输入 values 的形状: {values.shape}")

    with torch.autograd.set_detect_anomaly(True):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            values=values
        )
    print(outputs)

    # 6) 取 CLS token embedding
    # 调试输出 keys
    print(f"[embed_cells DEBUG] model 输出 keys: {list(outputs.keys())}")

    # 安全地拿到 CLS 向量
    if "cell_emb" in outputs:
        # scGPT的自定义输出，已经是在CLS位置做了 pooling
        cls_embedding = outputs["cell_emb"]     # [B, emb_dim]
    elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
        cls_embedding = outputs["last_hidden_state"][:, 0, :]
    else:
        # 万一 API 变动，兜底
        hidden = outputs.last_hidden_state
        cls_embedding = hidden[:, 0, :]

    print(f"[embed_cells DEBUG] 返回的 CLS 嵌入向量形状: {cls_embedding.shape}")
    return cls_embedding