import argparse
import os
import time
import numpy as np
import torch
from VAE_model import VAE
import sys
sys.path.append("..")
from guided_diffusion.cell_datasets_loader import load_data
torch.autograd.set_detect_anomaly(True)
import random
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_vae(args, state_dict=None):
    """Instantiates autoencoder and dataset to run an experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_data(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        train_vae=True,
    )

    autoencoder = VAE(
        num_genes=args["num_genes"],
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
    )
    if state_dict is not None:
        print('loading pretrained model from: \n',state_dict)
        use_gpu = device == "cuda"
        autoencoder.encoder.load_state(state_dict["encoder"], use_gpu)
        autoencoder.decoder.load_state(state_dict["decoder"], use_gpu)

    return autoencoder, datasets

def train_vae(args, return_model=False):
    """Train a autoencoder"""
    print(f"Loading data from: {args['data_dir']}")
    print("[Debug] - loading pretrained model from: \n", args["state_dict"])
    filenames = {}
    checkpoint_path = {
        "encoder": os.path.join(
            args["state_dict"], filenames.get("model", "encoder.ckpt")
        ),
        "decoder": os.path.join(
            args["state_dict"], filenames.get("model", "decoder.ckpt")
        ),
        "gene_order": os.path.join(
            args["state_dict"], filenames.get("gene_order", "gene_order.tsv")  # 2000 -> 1958
        ),
    }
    autoencoder, datasets = prepare_vae(args, checkpoint_path)
   
    args["hparams"] = autoencoder.hparams

    start_time = time.time()

    for step in tqdm(range(args["max_steps"]), desc="Training VAE", total=args["max_steps"]):
        genes, _ = next(datasets)
        genes = genes.float()
        # print(f"输入数据 'genes' 的数据类型是: {genes.dtype}")
        # print(f"模型 'autoencoder' 参数的数据类型是: {next(autoencoder.parameters()).dtype}")
        minibatch_training_stats = autoencoder.train_step(genes)

        if step % 1000 == 0:
            for key, val in minibatch_training_stats.items():
                print('step ', step, 'loss ', val)

        ellapsed_minutes = (time.time() - start_time) / 60

        stop = ellapsed_minutes > args["max_minutes"] or (
            step == args["max_steps"] - 1
        )

        if ((step % args["checkpoint_freq"]) == 0 or stop):

            os.makedirs(args["save_dir"],exist_ok=True)
            torch.save(
                autoencoder.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_step={}.pt".format(args["seed"], step),
                ),
            )

            if stop:
                break

    if return_model:
        return autoencoder, datasets

def parse_arguments():
    """Read arguments if this script is called from a terminal."""

    parser = argparse.ArgumentParser(description="Finetune Scimilarity")
    # dataset arguments
    parser.add_argument("--data_dir", type=str, default='../../../data/scrna_data/scrna_positive.h5ad')
    parser.add_argument("--loss_ae", type=str, default="mse")
    parser.add_argument("--decoder_activation", type=str, default="ReLU")

    # AE arguments                                             
    parser.add_argument("--local_rank", type=int, default=0)  
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--num_genes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="")

    # training arguments
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--state_dict", type=str, default="../../../checkpoints/scimilarity/model_v1.1")  # if pretrain
    # parser.add_argument("--state_dict", type=str, default=None)   # if not pretrain

    parser.add_argument("--save_dir", type=str, default='../../../checkpoints/scdiffusion/vae_checkpoint/VAE')
    parser.add_argument("--sweep_seeds", type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    seed_everything(1234)
    args = parse_arguments()
    print("Parsed arguments:", args)

    # Check if the final trained model already exists.
    # The final step is max_steps - 1.
    final_step = args["max_steps"] - 1
    final_model_path = os.path.join(
        args["save_dir"],
        f"model_seed={args['seed']}_step={final_step}.pt"
    )

    if os.path.exists(final_model_path):
        print(f"Found final pre-trained model at '{final_model_path}'. Skipping training.")
    else:
        print(f"Final model not found at '{final_model_path}'. Starting training.")
        train_vae(args)