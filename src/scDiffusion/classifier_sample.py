import argparse
import sys, os
import numpy as np
import pandas as pd
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import scanpy as sc
import matplotlib.pyplot as plt
from collections import defaultdict

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.scDiffusion.guided_diffusion import dist_util, logger
from src.scDiffusion.guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from utils.metrics import (
    compute_mae,
    compute_des,
    compute_pds,
    compute_edistance,
    compute_r2,
    compute_mmd,
    compute_pearson,
    compute_pearson_delta,
    compute_pearson_delta_de,
)
from VAE.VAE_model import VAE

def get_de_genes_from_anndatas(adata_ctrl, adata_pert, group_name='group', ctrl_name='Control', pert_name='Perturbed', pval_cutoff=0.05):
    """
    Helper function to find Differentially Expressed Genes by comparing two AnnData objects.
    """
    combined_adata = adata_ctrl.concatenate(adata_pert, batch_key=group_name, batch_categories=[ctrl_name, pert_name])
    sc.tl.rank_genes_groups(combined_adata, groupby=group_name, method='wilcoxon', groups=[pert_name], reference=ctrl_name)
    result = combined_adata.uns['rank_genes_groups']
    de_mask = result['pvals_adj'][pert_name] < pval_cutoff
    de_genes = set(np.array(result['names'][pert_name])[de_mask])
    all_genes = result['names'][pert_name]
    all_lfcs = result['logfoldchanges'][pert_name]
    gene_fold_changes = dict(zip(all_genes, all_lfcs))
    return de_genes, gene_fold_changes

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
            logger.log(f"Loading initial cells from: {args.init_cell_path}")
            ori_adata = sc.read_h5ad(args.init_cell_path)
            sc.pp.normalize_total(ori_adata, target_sum=1e4)
            sc.pp.log1p(ori_adata)

            label_map = {0: 'Control', 1: 'IFN'}

            start_condition_str = label_map[cell_type[0]]
            logger.log(f"Selecting starting cells with 'perturbation_status' == '{start_condition_str}'")

            adata = ori_adata[ori_adata.obs['perturbation_status'] == start_condition_str].copy()
            
            if adata.n_obs == 0:
                raise ValueError(f"No cells found with perturbation_status '{start_condition_str}' in the h5ad file.")
            
            logger.log(f"Found {adata.n_obs} starting cells.")

            start_x = adata.X.toarray() if isinstance(adata.X, np.memmap) or 'sparse' in str(type(adata.X)) else adata.X
            autoencoder = load_VAE(args.ae_dir, args.num_gene)
            autoencoder.eval()
            with th.no_grad():
                start_x = autoencoder(th.tensor(start_x, device=dist_util.dev(), dtype=th.float32), return_latent=True).detach().cpu().numpy()

            n, m = start_x.shape
            if n >= args.batch_size:
                indices = np.random.choice(n, args.batch_size, replace=False)
                start_x = start_x[indices, :]
            else:
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
            sample_num = sum(s.shape[0] for s in all_cell)
            logger.log(f"created {sample_num} samples")

    arr = np.concatenate(all_cell, axis=0)
    if arr.shape[0] > args.num_samples:
        arr = arr[:args.num_samples]
        
    os.makedirs(args.sample_dir, exist_ok=True)
    save_data(arr, traj, args.sample_dir+str(cell_type[0]))

    # --- Evaluation Module ---
    if inter: 
        logger.log("--- Starting Evaluation ---")

        # 1. Decode generated samples from latent to expression space
        logger.log("Decoding generated samples...")
        autoencoder = load_VAE(args.ae_dir, args.num_gene)
        autoencoder.to(dist_util.dev())
        autoencoder.eval()
        with th.no_grad():
            predicted_latent = th.tensor(arr, device=dist_util.dev(), dtype=th.float32)
            pred_X = autoencoder.decoder(predicted_latent).detach().cpu().numpy()

        # 2. Load and process ground truth data
        logger.log("Loading ground truth data for evaluation...")
        ori_adata = sc.read_h5ad(args.init_cell_path)
        sc.pp.normalize_total(ori_adata, target_sum=1e4)
        sc.pp.log1p(ori_adata)

        label_map = {0: 'Control', 1: 'IFN'}
        control_key = label_map[cell_type[0]]
        stim_key = label_map[cell_type[1]]

        ctrl_mask = ori_adata.obs['perturbation_status'] == control_key
        stim_mask = ori_adata.obs['perturbation_status'] == stim_key

        ctrl_adata_true = ori_adata[ctrl_mask].copy()
        stim_adata_true = ori_adata[stim_mask].copy()

        if ctrl_adata_true.n_obs < args.num_samples or stim_adata_true.n_obs < args.num_samples:
            raise ValueError(f"Not enough cells for evaluation. Need at least {args.num_samples}.")

        sc.pp.subsample(ctrl_adata_true, n_obs=args.num_samples, random_state=0)
        sc.pp.subsample(stim_adata_true, n_obs=args.num_samples, random_state=0)
        
        if not isinstance(ctrl_adata_true.X, np.ndarray):
            ctrl_X = ctrl_adata_true.X.toarray()
        else:
            ctrl_X = ctrl_adata_true.X

        if not isinstance(stim_adata_true.X, np.ndarray):
            true_pert_X = stim_adata_true.X.toarray()
        else:
            true_pert_X = stim_adata_true.X

        # Create AnnData for predicted cells
        pred_adata = sc.AnnData(pred_X, obs={'perturbation_status': [f'Predicted_{stim_key}'] * args.num_samples}, var=ctrl_adata_true.var)

        # 3. Calculate pseudobulk profiles
        true_pert_pb = np.mean(true_pert_X, axis=0)
        pred_pert_pb = np.mean(pred_X, axis=0)
        ctrl_pb = np.mean(ctrl_X, axis=0)

        # 4. Calculate all metrics
        logger.log("Calculating Differentially Expressed Genes (DEGs)...")
        true_de_genes, _ = get_de_genes_from_anndatas(ctrl_adata_true, stim_adata_true, pert_name='True_Stim')
        pred_de_genes, pred_gene_fold_changes = get_de_genes_from_anndatas(ctrl_adata_true, pred_adata, pert_name='Pred_Stim')
        logger.log(f"Found {len(true_de_genes)} true DEGs and {len(pred_de_genes)} predicted DEGs.")

        mae_val = compute_mae(true_pert_pb, pred_pert_pb)
        des_val = compute_des(true_de_genes, pred_de_genes, pred_gene_fold_changes)
        pds_val = compute_pds(pred_pert_pb.reshape(1,-1), true_pert_pb.reshape(1,-1))
        edist_val = compute_edistance(true_pert_X, pred_X)
        mmd_val = compute_mmd(true_pert_X, pred_X)
        r2_val = compute_r2(true_pert_X, pred_X)
        p_all = compute_pearson(true_pert_pb, pred_pert_pb)
        pd_all = compute_pearson_delta(true_pert_pb, pred_pert_pb, ctrl_pb)
        pd_de20 = compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=20)
        pd_de50 = compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=50)
        pd_de100 = compute_pearson_delta_de(true_pert_pb, pred_pert_pb, ctrl_pb, k=100)

        # 5. Print results
        logger.log(f"\nPerturbation Discrimination Score (PDS): {pds_val:.4f} (Note: 1.0 is expected for single perturbation eval)")
        logger.log(f"Mean Absolute Error (MAE): {mae_val:.4f}")
        logger.log(f"Differential Expression Score (DES): {des_val:.4f}")
        logger.log("-" * 20)
        logger.log(f"E-Distance: {edist_val:.4f}")
        logger.log(f"Maximum Mean Discrepancy (MMD): {mmd_val:.4f}")
        logger.log(f"R-squared (R2): {r2_val:.4f}")
        logger.log("-" * 20)
        logger.log(f"Pearson (all genes): {p_all:.4f}")
        logger.log(f"Pearson Delta (all genes): {pd_all:.4f}")
        logger.log(f"Pearson Delta (top 20 DE genes): {pd_de20:.4f}")
        logger.log(f"Pearson Delta (top 50 DE genes): {pd_de50:.4f}")
        logger.log(f"Pearson Delta (top 100 DE genes): {pd_de100:.4f}")
        logger.log("="*50 + "\nEvaluation complete!")

        # 6. UMAP Visualization
        logger.log("\n--- Generating UMAP Visualization ---")
        
        logger.log(f"Loading training data from: {args.train_data_path}")
        adata_train = sc.read_h5ad(args.train_data_path)

        # `ori_adata` is the full test set, loaded earlier.
        # `pred_adata` is the generated data.
        
        # Assign data sources
        adata_train.obs['data_source'] = 'train'
        ori_adata.obs['data_source'] = 'test'
        pred_adata.obs['data_source'] = 'generated'

        # Ensure unique cell names before concatenation
        adata_train.obs_names_make_unique()
        ori_adata.obs_names_make_unique()
        pred_adata.obs_names_make_unique()

        # Combine all data for visualization
        adata_viz = sc.concat(
            [adata_train, ori_adata, pred_adata],
            join='outer',
            index_unique=None,
            fill_value=0
        )
        
        # UMAP computation
        sc.pp.neighbors(adata_viz, n_neighbors=15, use_rep='X')
        sc.tl.umap(adata_viz, random_state=0)
        
        # --- Create a categorical column for plotting ---
        # 1. Define plot groups
        adata_viz.obs['plot_group'] = 'All Cells'
        
        # 2. Identify and label true perturbed and generated cells
        is_true_pert_mask = (adata_viz.obs['perturbation_status'] != 'Control') & (adata_viz.obs['data_source'] == 'test')
        adata_viz.obs.loc[is_true_pert_mask, 'plot_group'] = 'True Perturbed'
        
        is_generated_mask = adata_viz.obs['data_source'] == 'generated'
        adata_viz.obs.loc[is_generated_mask, 'plot_group'] = 'Generated Perturbed'

        # 3. Define the plotting order and colors
        category_order = [
            'All Cells',
            'True Perturbed',
            'Generated Perturbed'
        ]
        color_palette = {
            'All Cells': 'lightgray',
            'True Perturbed': 'blue',
            'Generated Perturbed': 'orange'
        }
        
        # 4. Make the column categorical with the specified order
        adata_viz.obs['plot_group'] = pd.Categorical(adata_viz.obs['plot_group'], categories=category_order)

        # 5. Plot using sc.pl.umap
        fig, ax = plt.subplots(figsize=(8, 6))
        point_size = 10
        
        sc.pl.umap(
            adata_viz,
            color='plot_group',
            palette=color_palette,
            ax=ax,
            size=point_size,
            title="", # Remove title
            legend_loc='best', # Place legend inside the plot
            show=False # We will handle saving and closing manually
        )
        
        os.makedirs(os.path.dirname(args.umap_plot), exist_ok=True)
        plt.savefig(args.umap_plot, dpi=300, bbox_inches='tight')
        logger.log(f"✔ Saved UMAP plot to {args.umap_plot}")
        plt.close(fig)
        # 7. Save Output
        os.makedirs(os.path.dirname(args.out_h5ad), exist_ok=True)
        pred_adata.write_h5ad(args.out_h5ad)
        logger.log(f"✔ Saved synthetic AnnData to {args.out_h5ad}")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser(celltype=[0], weight=[10,10]):
    defaults = dict(
        clip_denoised=True,
        batch_size=4,
        use_ddim=False,
        class_cond=False, 
        num_class=2, 
        classifier_scale1=weight[0]*2/10,
        classifier_scale2=weight[1]*2/10,
        init_time = 600,    # initial noised state if interpolation
        start_guide_steps = 500,     # the time to use classifier guidance
        filter = False,   # filter the simulated cells that are classified into other condition, might take long time
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the diffusion model checkpoint.")
    parser.add_argument("--classifier_path", type=str, required=True,
                        help="Path to the classifier checkpoint.")
    parser.add_argument("--ae_dir", type=str, required=True,
                        help="Path to the autoencoder checkpoint.")
    parser.add_argument("--num_gene", type=int, required=True,
                        help="Number of genes to use.")
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Directory to save generated samples.")
    parser.add_argument("--init_cell_path", type=str, required=True,
                        help="Path to the initial cell state file (.h5ad).")
    parser.add_argument("--train-data-path", type=str, required=True,
                        help="Path to the training AnnData file (.h5ad) for UMAP visualization.")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--out_h5ad", type=str, default="results/scdiffusion/synthetic_prediction.h5ad")
    parser.add_argument('--umap_plot', type=str, default="results/scdiffusion/umap_comparison.png")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    CONTROL_LABEL = 0
    IFN_LABEL = 1

    main(
        cell_type=[CONTROL_LABEL, IFN_LABEL], 
        inter=True, 
        weight=[0, 10]
    )

    print("\n--- Perturbation task complete. ---")
    print(f"Generated cells simulating 'IFN' state are saved in the configured output directory.")