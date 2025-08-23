# -*- coding: utf-8 -*-

import io
import os
import socket

import torch as th
import torch.distributed as dist
import argparse
from datetime import datetime
from Squidiff import dist_util,logger

from Squidiff.scrna_datasets import prepared_data
from Squidiff.resample import create_named_schedule_sampler
from Squidiff.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from Squidiff.train_util import TrainLoop, plot_loss

GPUS_PER_NODE = 1  # Set this to the actual number of GPUs per node

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    chunk_size = 2 ** 30  # Size limit for data chunks
    if dist.get_rank() == 0:
        with open(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        dist.broadcast(th.tensor(num_chunks), 0)
        for i in range(0, len(data), chunk_size):
            dist.broadcast(th.tensor(data[i: i + chunk_size]), 0)
    else:
        num_chunks = dist.broadcast(th.tensor(0), 0).item()
        data = bytes()
        for _ in range(num_chunks):
            chunk = th.zeros(chunk_size, dtype=th.uint8)
            dist.broadcast(chunk, 0)
            data += bytes(chunk.numpy())

    return th.load(io.BytesIO(data), **kwargs)

def run_training(args):
    # logger.configure is now called in the main execution block
    logger.log("*********creating model and diffusion**********")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = prepared_data(
        data_dir = args['data_path'],
        batch_size = args['batch_size'],
        use_drug_structure= args['use_drug_structure'],
        comb_num = args['comb_num']
    )
    
    start_time = datetime.now()
    logger.log(f'**********training started at {start_time} **********')
    train_ = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args['batch_size'],
        microbatch=args['microbatch'],
        lr=args['lr'],
        ema_rate=args['ema_rate'],
        log_interval=args['log_interval'],
        save_interval=args['save_interval'],
        resume_checkpoint=args['resume_checkpoint'],
        use_fp16=args['use_fp16'],
        fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],
        use_drug_structure= args['use_drug_structure'],
        comb_num=args['comb_num']
    )
    train_.run_loop()
    
    end_time = datetime.now()

    during_time = (end_time-start_time).seconds/60

    logger.log(f'start time: {start_time} end_time: {end_time} time:{during_time} min')
    
    return train_.loss_list


def parse_args():
    """Parse command-line arguments and update with default values."""
    # Define default arguments
    default_args = {}
    default_args.update(model_and_diffusion_defaults())
    updated_args = {
        'data_path': '',
        'schedule_sampler': 'uniform',
        'lr': 1e-4,
        'weight_decay': 0.0,
        'lr_anneal_steps': 1e5,
        'batch_size': 128,
        'microbatch': -1,
        'ema_rate': '0.9999',
        'log_interval': 1e4,
        'save_interval': 1e4,
        'resume_checkpoint': '',
        'use_fp16': False,
        'fp16_scale_growth': 1e-3,
        'gene_size': 100,
        'output_dim': 100,
        'num_layers': 3,
        'class_cond': False,
        'use_encoder': True,
        'diffusion_steps': 1000,
        'logger_path': '',
        'use_drug_structure':False,
        'comb_num':1
    }
    default_args.update(updated_args)
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Perturbation-conditioned generative diffusion model')
    
    # Add arguments to the parser (these should correspond to the keys in default_args)
    for key, value in default_args.items():
        # A simple way to handle boolean arguments from strings
        arg_type = type(value)
        if arg_type == bool:
            # This allows --use_fp16 True or --use_fp16 False
            arg_type = lambda x: (str(x).lower() == 'true')
        parser.add_argument(f'--{key}', default=value, type=arg_type, help=f'{key} (default: {value})')

    # Parse command-line arguments
    args = parser.parse_args()

    # Convert the parsed arguments to a dictionary
    updated_args = vars(args)
    
    # Check if 'logger_path' is provided
    if not updated_args['logger_path']:
        print('ERROR: Please specify the logger path --logger_path.')
        raise ValueError("Logger path is required. Please specify the logger path.")

    # Check if 'data_path' is provided
    if not updated_args['data_path']:
        print("ERROR: Please specify the data path --data_path.")
        raise ValueError("Dataset path is required. Please specify the path where the training adata is.")

    # Return the updated arguments as a dictionary
    return updated_args



if __name__ == "__main__":
    args_train = parse_args()
    print('**************training args*************')
    print(args_train)

    # Configure the logger here, so it's available for all messages.
    logger.configure(dir=args_train['logger_path'])

    # Check if the final trained model already exists to avoid re-training.
    final_step = int(args_train['lr_anneal_steps'])
    
    final_model_name = f"model.pt"
    final_model_path = os.path.join(args_train['resume_checkpoint'], final_model_name)

    if os.path.exists(final_model_path):
        logger.log(f"Found pre-trained model at '{final_model_path}'. Skipping training.")
    else:
        logger.log(f"Final model checkpoint not found at '{final_model_path}'. Starting training.")
        losses = run_training(args_train)
        plot_loss(losses, args_train)

