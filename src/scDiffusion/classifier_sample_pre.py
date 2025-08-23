"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import scipy.sparse
import pandas as pd
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import scanpy as sc
import torch
from VAE.VAE_model import VAE

def load_VAE(ae_dir, num_gene):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(ae_dir))
    return autoencoder

def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main(cell_type=[0], multi=False, inter=False, weight=[10,10]):
    args = create_argparser(cell_type, weight).parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading classifier...")
    if multi:
        args.num_class = args.num_class1 # how many classes in this condition
        classifier1 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier1.load_state_dict(
            dist_util.load_state_dict(args.classifier_path1, map_location="cpu")
        )
        classifier1.to(dist_util.dev())
        classifier1.eval()

        args.num_class = args.num_class2 # how many classes in this condition
        classifier2 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier2.load_state_dict(
            dist_util.load_state_dict(args.classifier_path2, map_location="cpu")
        )
        classifier2.to(dist_util.dev())
        classifier2.eval()

    else:
        classifier = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        classifier.eval()

    '''
    control function for Gradient Interpolation Strategy
    '''
    def cond_fn_inter(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        # xt = diffusion.q_sample(th.tensor(init,device=dist_util.dev()),t*th.ones(init.shape[0],device=dist_util.dev(),dtype=torch.long),)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected1 = log_probs[range(len(logits)), y1.view(-1)]
            selected2 = log_probs[range(len(logits)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2

            # l2_loss = ((x_in-xt)**2).mean()
            # grad3 = th.autograd.grad(-l2_loss, x_in, retain_graph=True)[0] * 100

            return grad1+grad2#+grad3

    '''
    control function for multi-conditional generation
    Two conditional generation here
    '''
    def cond_fn_multi(x, t, y=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits1 = classifier1(x_in, t)
            log_probs1 = F.log_softmax(logits1, dim=-1)
            selected1 = log_probs1[range(len(logits1)), y1.view(-1)]

            logits2 = classifier2(x_in, t)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            selected2 = log_probs2[range(len(logits2)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2
            
            return grad1+grad2

    '''
    control function for one conditional generation
    '''
    def cond_fn_ori(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in, retain_graph=True)[0] * args.classifier_scale
            return grad
        
    def model_fn(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        if args.class_cond:
            return model(x, t, y if args.class_cond else None)
        else:
            return model(x, t)
        
    if inter:
        # input real cell expression data as initial noise
        ori_adata = sc.read_h5ad(args.init_cell_path)
        sc.pp.normalize_total(ori_adata, target_sum=1e4)
        sc.pp.log1p(ori_adata)

    logger.log("sampling...")
    all_cell = []
    sample_num = 0
    while sample_num < args.num_samples:
        model_kwargs = {}

        if not multi and not inter:
            classes = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)

        if multi:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            # classes3 = ... if more conditions
            classes = th.stack((classes1,classes2), dim=1)

        if inter:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes = th.stack((classes1,classes2), dim=1)

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        if inter:

            # 1. 定义 Scimilarity 模型和基因顺序文件的路径
            scimilarity_model_dir = "../../checkpoints/scimilarity/model_v1.1" # 确保这个路径正确
            gene_order_path = os.path.join(scimilarity_model_dir, "gene_order.tsv")

            if not os.path.exists(gene_order_path):
                raise FileNotFoundError(f"Gene order file not found at: {gene_order_path}")
                
            # 2. 读取1958个基因的列表
            with open(gene_order_path, "r") as f:
                gene_order = [line.strip() for line in f]

            logger.log(f"Loading initial cells from: {args.init_cell_path}")
            ori_adata = sc.read_h5ad(args.init_cell_path)
            
            # 3. 在预处理前，根据加载的基因顺序过滤您的 AnnData 对象
            # 确保您的 AnnData 对象的 .var_names 是基因名
            logger.log(f"Subsetting data from {ori_adata.n_vars} genes to {len(gene_order)} genes.")
            
            # 检查您的数据中存在多少个预训练的基因
            available_genes = [gene for gene in gene_order if gene in ori_adata.var_names]
            if len(available_genes) < len(gene_order):
                print(f"Warning: Only {len(available_genes)} out of {len(gene_order)} pretrained genes were found in your data.")
            
            # 使用 .reindex() 来对齐基因，缺失的基因会用 0 填充
            ori_adata_sub = ori_adata[:, available_genes].copy()

            if scipy.sparse.issparse(ori_adata_sub.X):
                # 如果是稀疏矩阵，则转换为密集数组
                data_for_df = ori_adata_sub.X.toarray()
            else:
                # 如果已经是密集数组，则直接使用
                data_for_df = ori_adata_sub.X

            df = pd.DataFrame(
                data_for_df, 
                index=ori_adata_sub.obs_names, 
                columns=ori_adata_sub.var_names
            )

            # 使用 Pandas 强大的 reindex 功能来对齐基因。
            # 这会自动添加 gene_order 中有但 df 中没有的基因作为全零列
            df_reindexed = df.reindex(columns=gene_order, fill_value=0.0)

            # 从 reindexed DataFrame 创建最终的 AnnData 对象，保留原始的 obs 信息
            ori_adata = sc.AnnData(df_reindexed, obs=ori_adata.obs)

            # 确认基因数量现在与 gene_order 列表的长度一致
            assert ori_adata.n_vars == len(gene_order)

            sc.pp.normalize_total(ori_adata, target_sum=1e4)
            sc.pp.log1p(ori_adata)

            # 定义标签映射关系，0 -> Control, 1 -> IFN
            label_map = {0: 'Control', 1: 'IFN'}
            
            # 根据输入的cell_type[0]（即起始状态的标签）获取其字符串名称
            start_condition_str = label_map[cell_type[0]]
            logger.log(f"Selecting starting cells with 'perturbation_status' == '{start_condition_str}'")

            # 使用 'perturbation_status' 列进行筛选，获取所有 'Control' 细胞
            adata = ori_adata[ori_adata.obs['perturbation_status'] == start_condition_str].copy()
            
            if adata.n_obs == 0:
                raise ValueError(f"No cells found with perturbation_status '{start_condition_str}' in the h5ad file.")
            
            logger.log(f"Found {adata.n_obs} starting cells.")

            start_x = adata.X.toarray() if isinstance(adata.X, np.memmap) or 'sparse' in str(type(adata.X)) else adata.X
            autoencoder = load_VAE(args.ae_dir, args.num_gene)
            autoencoder.eval() # 确保 VAE 在评估模式
            with th.no_grad(): # 推理时不需要计算梯度
                start_x = autoencoder(th.tensor(start_x, device=dist_util.dev(), dtype=th.float32), return_latent=True).detach().cpu().numpy()

            n, m = start_x.shape
            if n >= args.batch_size:
                # 如果起始细胞数足够，则取 batch_size 个
                indices = np.random.choice(n, args.batch_size, replace=False)
                start_x = start_x[indices, :]
            else:
                # 如果不够，则重复采样以凑够 batch_size
                indices = np.random.choice(n, args.batch_size, replace=True)
                start_x = start_x[indices, :]
            
            noise = diffusion.q_sample(th.tensor(start_x,device=dist_util.dev()),args.init_time*th.ones(start_x.shape[0],device=dist_util.dev(),dtype=torch.long),)
            model_kwargs["init"] = start_x
            model_kwargs["diffusion"] = diffusion

        if multi:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_multi,
                device=dist_util.dev(),
                noise = None,
                start_time=diffusion.betas.shape[0],
                start_guide_steps=args.start_guide_steps,
            )
        elif inter:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_inter,
                device=dist_util.dev(),
                noise = noise,
                start_time=diffusion.betas.shape[0],
                start_guide_steps=args.start_guide_steps,
            )
        else:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_ori,
                device=dist_util.dev(),
                noise = None,
            )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        if args.filter:
            for sample in gathered_samples:
                if multi:
                    logits1 = classifier1(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    logits2 = classifier2(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    prob1 = F.softmax(logits1, dim=-1)
                    prob2 = F.softmax(logits2, dim=-1)
                    type1 = torch.argmax(prob1, 1)
                    type2 = torch.argmax(prob2, 1)
                    select_index = ((type1 == cell_type[0]) & (type2 == cell_type[1]))
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
                elif inter:
                    logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    prob = F.softmax(logits, dim=-1)
                    left = (prob[:,cell_type[0]] > weight[0]/10-0.15) & (prob[:,cell_type[0]] < weight[0]/10+0.15)
                    right = (prob[:,cell_type[1]] > weight[1]/10-0.15) & (prob[:,cell_type[1]] < weight[1]/10+0.15)
                    select_index = left & right
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
                else:
                    logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    prob = F.softmax(logits, dim=-1)
                    predicted_type  = torch.argmax(prob, 1)
                    select_index = (predicted_type  == cell_type[0])
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
            logger.log(f"created {sample_num} samples")
        else:
            all_cell.extend([sample.cpu().numpy() for sample in gathered_samples])
            sample_num = len(all_cell) * args.batch_size
            logger.log(f"created {len(all_cell) * args.batch_size} samples")

    arr = np.concatenate(all_cell, axis=0)
    save_data(arr, traj, args.sample_dir+str(cell_type[0]))

    dist.barrier()
    logger.log("sampling complete")


def create_argparser(celltype=[0], weight=[10,10]):
    defaults = dict(
        clip_denoised=True,
        num_samples=1000,
        batch_size=500,
        use_ddim=False,
        class_cond=False, 

        model_path="../../checkpoints/scdiffusion/diffusion_checkpoint/my_diffusion/model000000.pt", 
        classifier_path="../../checkpoints/scdiffusion/classifier_checkpoint/2-classifier/model100000.pt",
        ae_dir='../../checkpoints/scdiffusion/vae_checkpoint/VAE/model_seed=0_step=150000.pt', 
        num_gene=1958,
        num_class=2, 
        sample_dir=f"../../results/scdiffusion/simulated_samples/",

        classifier_scale1=weight[0]*2/10,
        classifier_scale2=weight[1]*2/10,
        init_time = 600,    # initial noised state if interpolation
        init_cell_path = '../../data/scrna_data/scrna_positive.h5ad',   #input initial noised cell state
        start_guide_steps = 500,     # the time to use classifier guidance
        filter = False,   # filter the simulated cells that are classified into other condition, might take long time
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # --- 1. 定义您的类别标签 ---
    # 确保这与您训练分类器时使用的整数标签一致
    CONTROL_LABEL = 0
    IFN_LABEL = 1

    # --- 2. 调用 main 函数执行扰动任务 ---
    # cell_type: [起始状态标签, 目标状态标签]
    # inter=True: 激活梯度插值/扰动模式
    # weight: [起始权重, 目标权重]。我们想完全变成IFN状态，所以Control权重为0，IFN权重为10（最大）。
    main(
        cell_type=[CONTROL_LABEL, IFN_LABEL], 
        inter=True, 
        weight=[0, 10]
    )

    print("\n--- Perturbation task complete. ---")
    print(f"Generated cells simulating 'IFN' state are saved in the configured output directory.")