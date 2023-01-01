################################ IMPORTS ################################################################################
import dnnlib
from dnnlib.util import EasyDict
from utils import *
import wandb
import os
import click

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import copy
import sys

import math
import time
import gc
import traceback
import itertools

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from networks.stylegan import LipEqualLinear, StyleGANDiscriminator
from networks.custom_modules import MultiShapeStyleLinearApproach, MyRegularizer
from training.dataset import ImageFolderDataset
from torch_utils.misc import InfiniteSampler
from metrics import metric_main

################################################################################################################

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def get_metric(metric, generator, c):
    dataset = get_dataset(c)
    dataset_kwargs = EasyDict()
    dataset_kwargs.dataset = dataset
    dataset_kwargs.name = 'TempName'

    class ModelWrapper(nn.Module):
        def __init__(self, generator, c):
            super().__init__()
            self.generator = generator
            self.c = c

            self.z_dim = self.generator.hidden_dim
            self.c_dim = 0

        def forward(self, z, c, **kwargs):
            input_noise = None
            generator_output = self.generator(z, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='decode')
            output = generator_output[f'renders/{self.c.steps}/render']
            return output

    generator = ModelWrapper(generator, c)
    
    return metric_main.calc_metric(metric, G=generator, num_gpus=c.world_size, rank=c.rank, device=c.device, dataset_kwargs=dataset_kwargs)


# ------------------------------------------------------------------------------------------

def log_wandb(x, c):
    if c.rank == 0:
        wandb.log(x, step=c.global_step, commit=False)

def get_dataset(c):
    shape_transform = transforms.Compose([
        transforms.Resize(c.G_args.output_size),
        transforms.CenterCrop(c.G_args.output_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # data = CustomImageFolder(
    #     c.data_path,
    #     transform=shape_transform,
    # )
    
    dataset = torchvision.datasets.LSUN(
        '/scratch/ssd002/datasets/LSUN/lsun', ['bedroom_train'], 
        transform=shape_transform,
    )

    if len(dataset) > 200_000:
        dataset = torch.utils.data.Subset(dataset, list(range(200_000)))
    
    return dataset

def get_dataloader(c):
    dataset = get_dataset(c)

    sampler = DistributedSampler(dataset, drop_last=True, num_replicas=c.world_size, rank=c.rank)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=c.batch_size // c.world_size, sampler=sampler, num_workers=c.dataloader_workers, pin_memory=True
    )
    
    return dataloader

def training_loop(c):
    torch.backends.cudnn.benchmark = True               # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.

    print(torch.cuda.nccl.version(), flush=True)

    # Instantiate generator and discriminator
    generator = MultiShapeStyleLinearApproach(**c.G_args).to(c.device)
    generator_ema = copy.deepcopy(generator).eval()

    # Resume from checkpoint
    if (c.resume is not None) and (c.rank == 0):
        if not os.path.exists(c.resume):
            print(f'Path to resume doesnt exists. Starting from scratch...')
        else:
            print(f'Resuming from "{c.resume}"')
            loaded_dict = torch.load(c.resume)
            c.global_step = loaded_dict['global_step']
            generator.load_state_dict(loaded_dict['generator'])
            generator_ema.load_state_dict(loaded_dict['generator_ema'])
            c.ema_rampup = None # Disable ema rampup
    
    # Distribute across GPUs.
    if c.rank == 0:
        print(f'Distributing across {c.world_size} GPUs...')
    if c.world_size > 1:
        for module in [generator, generator_ema]:
            if module is not None:
                for param in params_and_buffers(module):
                    dist.broadcast(param, src=0)
    

    print(get_metric('fid50k_full', generator_ema, c), flush=True)

    
################################################################################################################


@click.command()
# Required.
@click.option('--save-dir',           help='Where to save the results', metavar='DIR',                           required=True)
@click.option('--data',               help='Path to training data folder',                                       type=str, required=True)
@click.option('--batch',              help='Total batch size', metavar='INT',                                    type=click.IntRange(min=1), required=True)
@click.option('--output-size',        help='Resolution of samples generated by the generator',                   type=click.Choice(['64', '128', '256']), required=True)
@click.option('--feature-volume-size',help='Resolution of the feature volume', metavar='INT',                    type=click.IntRange(min=16), required=True)
@click.option('--num-batch-splits',   help='Number of batches to accumulate per step', metavar='INT',            type=click.IntRange(min=1), required=True)
@click.option('--d-reg-every',        help='Dicsriminator regularization every n batches', metavar='INT',        type=click.IntRange(min=1), required=True)
@click.option('--r1-weight',          help='R1 regularization weight', metavar='FLOAT',                          type=click.FloatRange(min=0), required=True)
@click.option('--g-reg-every',        help='Generator regularization every n batches', metavar='INT',            type=click.IntRange(min=1), required=True)
@click.option('--t1-weight',          help='T1 regularization weight', metavar='FLOAT',                          type=click.FloatRange(min=0), required=True)
@click.option('--random-flip-prob',   help='Probability to flip the label of an input to the discriminator',  metavar='FLOAT',  type=click.FloatRange(min=0, max=1), required=True)
@click.option('--random-flip-kimg',   help='Number of images in the beginning to perform random label flips', metavar='INT',    type=click.IntRange(min=0), required=True)

# Optional features.
@click.option('--resume',             help='Resume from given network pickle',  metavar='[PATH|URL]',  type=str)
@click.option('--entropy-weight',     help='Entropy regularization weight',     metavar='FLOAT',        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--non-useless-weight', help='Non useless regularization weight', metavar='FLOAT',        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--non-empty-weight',   help='Non empty regularization weight',   metavar='FLOAT',        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--pl-weight',          help='Path length regularization weight', metavar='FLOAT',        type=click.FloatRange(min=0), default=2, show_default=True)
@click.option('--jacobian-weight',    help='Sprite unit jacobian weight',       metavar='FLOAT',        type=click.FloatRange(min=0), default=1, show_default=True)

@click.option('--complexity-max-weight',          help='Sprite complexity maximum reg weight', metavar='FLOAT',               type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--complexity-min-weight',          help='Sprite complexity minimum reg weight', metavar='FLOAT',               type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--complexity-anneal-start',        help='Sprite complexity reg annealing start (KIMG)', metavar='INT',         type=click.IntRange(min=0), default=50, show_default=True)
@click.option('--complexity-anneal-cycle',        help='Sprite complexity reg annealing cycle length (KIMG)', metavar='INT',  type=click.IntRange(min=0), default=500, show_default=True)
@click.option('--complexity-anneal-warmup',       help='Sprite complexity reg annealing warmup length (KIMG)', metavar='INT', type=click.IntRange(min=0), default=100, show_default=True)
@click.option('--complexity-anneal-gamma',        help='Sprite complexity reg annealing gamma', metavar='FLOAT',              type=click.FloatRange(min=0), default=1.0, show_default=True)

@click.option('--texture-complexity-mul',   help='Texture complexity multiplier vs sprite', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)




# Generator hyperparameters
@click.option('--hidden-dim',                           help='Generator hidden dim', metavar='INT',                        type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--hidden-action-dim',                    help='Generator hidden action dim', metavar='INT',                 type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--planner-type',                         help='Generator planner module type',                              type=click.Choice(['mlp', 'recurrent-1', 'recurrent-2', 'recurrent-3']), default='mlp', show_default=True)
@click.option('--shape-style-dim',                      help='Generator style dim', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--shape-input-dim',                      help='Generator dictionary size', metavar='INT',                   type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--shape-output-dim',                     help='Generator dictionary hidden dim', metavar='INT',             type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--to-rgb-type',                          help='Generator to-rgb module type',                               type=click.Choice(['none', 'simple', 'styled']), default='none', show_default=True)
@click.option('--shape-num-sinusoidals',                help='Number of frequencies for shapes', metavar='INT',            type=click.IntRange(min=0), default=4, show_default=True)
@click.option('--shape-progressive',                    help='Progressive frequency for shapes', metavar='BOOL',           type=bool, default=False, show_default=True)
@click.option('--texture-num-sinusoidals',              help='Number of frequencies for textures', metavar='INT',          type=click.IntRange(min=0), default=6, show_default=True)
@click.option('--texture-progressive',                  help='Progressive frequency for textures', metavar='BOOL',         type=bool, default=True, show_default=True)
@click.option('--use-textures',                         help='Use texture fields or not',          metavar='BOOL',         type=bool, default=True, show_default=True)
@click.option('--decoder-const-in',                     help='Use constant input in the StyleDecoder', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--decoder-size-in',                      help='The resolution of input in the StyleDecoder',                type=click.Choice(['2', '4', '8', '16', '32', '64']), default=4, show_default=True)
@click.option('--decoder-c-model',                      help='The width of the StyleDecoder',                              type=click.IntRange(min=16), default=32, show_default=True)
@click.option('--planner-lr-mul',                       help='Lr multiplier for planner', metavar='FLOAT',                 type=click.FloatRange(min=0.01), default=0.01, show_default=True)
@click.option('--shape-library-lr-mul',                 help='Lr multiplier for shape lib', metavar='FLOAT',               type=click.FloatRange(min=0.01), default=1.0, show_default=True)
@click.option('--texture-library-lr-mul',               help='Lr multiplier for texture lib', metavar='FLOAT',             type=click.FloatRange(min=0.01), default=0.01, show_default=True)


@click.option('--use-noise',                            help='Use per-pixel noise injections', metavar='BOOL',             type=bool, default=False, show_default=True)
@click.option('--clip-grad-norm',                       help='Gradient clipping (-1 for no clipping)', metavar='FLOAT',    type=click.FloatRange(min=-1), default=-1, show_default=True)



@click.option('--shape-progressive-unlock-kimg',        help='Number of KIMG needed to fully unlock shapes', metavar='INT',       
              type=click.IntRange(min=0), default=100, show_default=True)
@click.option('--texture-progressive-unlock-kimg',      help='Number of KIMG needed to fully unlock textures', metavar='INT',       
              type=click.FloatRange(min=0), default=1000, show_default=True)
@click.option('--progressive-start-unlock-kimg',        help='Number of KIMG needed to start unlocking progressives', metavar='INT',       
              type=click.IntRange(min=0), default=50, show_default=True)


@click.option('--steps',        help='Number of sprites', metavar='INT',                           type=click.IntRange(min=0), default=6, show_default=True)
# Misc hyperparameters.
@click.option('--glr',          help='G learning rate', metavar='FLOAT',                           type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                           type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT',    type=click.IntRange(min=1))
@click.option('--ema-kimg',     help='Half-life of the EMA of generator weights.', metavar='INT',  type=click.IntRange(min=0), default=10, show_default=True)
@click.option('--ema-rampup',   help='EMA ramp-up coefficient.', metavar='FLOAT',                  type=click.IntRange(min=0), default=0.05, show_default=True)

# Misc settings.
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=6, show_default=True)
@click.option('--plot',         help='How often to plot progress',  metavar='KIMG',             type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='KIMG',             type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)

def main(**kwargs):

    ################ Setup Config ################  
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.

    c.seed = opts.seed
    
    c.world_size = int(os.environ["SLURM_NTASKS"])
    c.rank = int(os.environ["SLURM_PROCID"])
    c.local_rank = int(os.environ["SLURM_LOCALID"])
    
    # print(json.dumps({x:y for x, y in dict(os.environ).items() if 'SLURM' in x}, sort_keys=True, indent=4), flush=True)
    # print(f"[LRank: {c.local_rank}, Rank: {c.rank}, World: {c.world_size}] GPU number {c.local_rank % torch.cuda.device_count()} on {os.environ['HOSTNAME']} is being binded to", flush=True)
    c.device = torch.device('cuda', c.local_rank % torch.cuda.device_count())
    if torch.cuda.is_available():
        torch.cuda.set_device(c.device)


    c.job_id = os.environ["SLURM_JOB_ID"]
    c.save_dir = opts.save_dir
    c.checkpoint_path = c.save_dir + '/checkpoint.zip'
    c.data_path = opts.data
    c.batch_size = opts.batch
    if c.batch_size % c.world_size != 0:
        raise ValueError(f"[SANITY CHECK FAILED] The batch size {c.batch_size} is not divisible by the number of replicas {c.world_size}")
    c.num_batch_splits = opts.num_batch_splits

    c.d_reg_every = opts.d_reg_every
    c.r1_weight = opts.r1_weight
    
    c.g_reg_every = opts.g_reg_every
    c.t1_weight = opts.t1_weight
    c.entropy_weight = opts.entropy_weight
    c.non_useless_weight = opts.non_useless_weight
    c.non_empty_weight = opts.non_empty_weight
    c.pl_weight = opts.pl_weight
    c.jacobian_weight = opts.jacobian_weight
    
    c.complexity_max_weight = opts.complexity_max_weight
    c.complexity_min_weight = opts.complexity_min_weight
    c.complexity_anneal_start = opts.complexity_anneal_start
    c.complexity_anneal_cycle = opts.complexity_anneal_cycle
    c.complexity_anneal_warmup = opts.complexity_anneal_warmup
    c.complexity_anneal_gamma = opts.complexity_anneal_gamma
    c.texture_complexity_mul = opts.texture_complexity_mul

    c.G_args = dnnlib.EasyDict()
    c.G_args.hidden_dim = opts.hidden_dim
    c.G_args.hidden_action_dim = opts.hidden_action_dim
    c.G_args.planner_type = opts.planner_type
    c.G_args.num_actions = opts.steps
    c.G_args.shape_style_dim = opts.shape_style_dim
    c.G_args.shape_encoding_dim = opts.shape_output_dim
    c.G_args.shape_progressive = opts.shape_progressive
    c.G_args.shape_num_sinusoidals = opts.shape_num_sinusoidals
    c.G_args.use_textures = opts.use_textures
    c.G_args.texture_progressive = opts.texture_progressive
    c.G_args.texture_num_sinusoidals = opts.texture_num_sinusoidals
    c.G_args.to_rgb_type = opts.to_rgb_type
    c.G_args.output_size = int(opts.output_size)
    c.feature_volume_size = int(opts.feature_volume_size)
    c.G_args.const_in = opts.decoder_const_in
    c.G_args.size_in = int(opts.decoder_size_in)
    c.G_args.c_model = opts.decoder_c_model
    c.G_args.planner_lr_mul = opts.planner_lr_mul
    c.G_args.shape_library_lr_mul = opts.shape_library_lr_mul
    c.G_args.texture_library_lr_mul = opts.texture_library_lr_mul


    c.steps = opts.steps
    c.shape_progressive_unlock_kimg = opts.shape_progressive_unlock_kimg
    c.texture_progressive_unlock_kimg = opts.texture_progressive_unlock_kimg
    c.progressive_start_unlock_kimg = opts.progressive_start_unlock_kimg
    c.ema_kimg = opts.ema_kimg
    c.ema_rampup = opts.ema_rampup
    c.use_noise = opts.use_noise
    c.clip_grad_norm = opts.clip_grad_norm

    c.generator_lr = opts.glr
    c.discriminator_lr = opts.dlr
    
    c.random_flip_prob = opts.random_flip_prob
    c.random_flip_kimg = opts.random_flip_kimg

    c.kimg = opts.kimg
    c.tick = opts.tick
    c.plot = opts.plot
    c.snap = opts.snap
    c.dataloader_workers = opts.workers

    c.global_step = 0
    c.resume = opts.resume
    
    # Setup distributed package
    if c.world_size > 1:
        dist.init_process_group("nccl", rank=c.rank, world_size=c.world_size)

    training_loop(c)

if __name__ == "__main__":
    try:
        main() 
    except:
        print("Something went wrong!")
        traceback.print_exc()
        wandb.finish()
        dist.destroy_process_group()
        sys.exit(1)


