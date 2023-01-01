import dnnlib
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

from networks.stylegan import LipEqualLinear, StyleGANDiscriminator
from networks.custom_modules import MultiShapeStyleLinearApproach, MyRegularizer
from training.dataset import ImageFolderDataset
from torch_utils.misc import InfiniteSampler
from utils import num_trainable_params
from metrics import metric_main

import importlib
foobar = importlib.import_module('train-gan-ddp')


G_args = dnnlib.EasyDict()
G_args.hidden_dim = 256
G_args.hidden_action_dim = 512
G_args.planner_type = 'mlp'
G_args.num_actions = 8
G_args.shape_style_dim = 256
G_args.shape_encoding_dim = 64
G_args.shape_progressive = False   
G_args.shape_num_sinusoidals = 4
G_args.use_textures = False
G_args.texture_progressive = False
G_args.texture_num_sinusoidals = 8
G_args.to_rgb_type = 'styled'
G_args.output_size = 64
feature_volume_size = 64
G_args.const_in = True
G_args.size_in = 4
G_args.c_model = 32
G_args.planner_lr_mul = 0.01
G_args.shape_library_lr_mul = 1.0
G_args.texture_library_lr_mul = 0.01


c = dnnlib.EasyDict()
c.G_args = G_args
c.feature_volume_size = 64
c.steps = 8
c.use_noise = False
c.data_path = 'temp'
c.world_size = int(os.environ["SLURM_NTASKS"])
c.rank = int(os.environ["SLURM_PROCID"])
c.local_rank = int(os.environ["SLURM_LOCALID"])

print(f"{c.rank}, {c.local_rank}, {c.world_size}", flush=True)

c.device = torch.device('cuda', c.local_rank % torch.cuda.device_count())
if torch.cuda.is_available():
    torch.cuda.set_device(c.device)

 # Setup distributed package
if c.world_size > 1:
    dist.init_process_group("nccl", rank=c.rank, world_size=c.world_size)

generator = MultiShapeStyleLinearApproach(**G_args).to(c.device)
if c.rank == 0:
    generator.load_state_dict(torch.load('/h/amsabour/checkpoint/AMSP1/8648195/checkpoint.zip')['generator_ema'])

if c.world_size > 1:
    for module in [generator]:
        if module is not None:
            for param in foobar.params_and_buffers(module):
                dist.broadcast(param, src=0)

print(foobar.get_metric('fid50k_full', generator, c), flush=True)








