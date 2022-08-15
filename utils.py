import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import os

import warnings



def is_rank_zero():
    return get_rank() == 0


def print_once(s):
    if is_rank_zero():
        print(s)


def get_rank():
    return int(os.environ.get('LOCAL_RANK', 0))



def compute_kl_against_standard_gaussian(mean, log_var):
    assert mean.shape[-1] == log_var.shape[-1]
    embed_size = mean.shape[-1]
    return 0.5 * (torch.sum(mean ** 2, dim=-1) + torch.sum(log_var.exp(), dim=-1) - embed_size - torch.sum(log_var, dim=-1))


class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.loaded_samples = dict()

    def __getitem__(self, index):
        if index in self.loaded_samples.keys():
            return self.loaded_samples[index]

        output = super().__getitem__(index)
        self.loaded_samples[index] = output
        return output
    
def create_linspace(begin=-1, end=1, H=64, W=64, as_center_points=False):
    if as_center_points:
        """
        Points are obtained by sampling the centers of a grid of HxW rectangles
        _________________________
        |     |     |     |     |
        |  *  |  *  |  *  |  *  |
        |_____|_____|_____|_____|
        |     |     |     |     |
        |  *  |  *  |  *  |  *  |
        |_____|_____|_____|_____|
        |     |     |     |     |
        |  *  |  *  |  *  |  *  |
        |_____|_____|_____|_____|
        |     |     |     |     |
        |  *  |  *  |  *  |  *  |
        |_____|_____|_____|_____|
        """
        x = torch.cat([
            torch.linspace(begin + (end - begin) / (2 * H), end - (end - begin) / (2 * H), steps=H).view(H, 1, 1).expand(H, W, 1), 
            torch.linspace(begin + (end - begin) / (2 * W), end - (end - begin) / (2 * W), steps=W).view(1, W, 1).expand(H, W, 1)
        ], dim=-1)
    else:
        """
        Points are obtained by sampling along the edges of a grid 
        
        *-----*-----*-----*-----*
        |     |     |     |     |
        |     |     |     |     |     
        *-----*-----*-----*-----*
        |     |     |     |     |
        |     |     |     |     |     
        *-----*-----*-----*-----*
        |     |     |     |     |
        |     |     |     |     |     
        *-----*-----*-----*-----*
        |     |     |     |     |
        |     |     |     |     |     
        *-----*-----*-----*-----*
        """
        x = torch.cat([
            torch.linspace(begin, end, steps=H).view(H, 1, 1).expand(H, W, 1), 
            torch.linspace(begin, end, steps=W).view(1, W, 1).expand(H, W, 1)
        ], dim=-1)
    
    return x # [H, W, 2]

def coords_to_sinusoidals(coords, num_sinusoidals=16, multiplier=2, freq_type='mult'):
    if freq_type not in ['mult', 'add']:
        raise RuntimeError("freq_type unknown. freq_type must be in ['mult', 'add']")
    
    sinusoidals = []
    for i in range(num_sinusoidals):
        if freq_type == 'mult':
            sinusoidals.append(torch.cos((multiplier ** i) * np.pi * coords))
            sinusoidals.append(torch.sin((multiplier ** i) * np.pi * coords))
        elif freq_type == 'add':
            sinusoidals.append(torch.cos((multiplier * (i + 1)) * np.pi * coords))
            sinusoidals.append(torch.sin((multiplier * (i + 1)) * np.pi * coords))
            
    output = torch.cat(sinusoidals, dim=-1)
    return output


def multiply_as_convex(a, b):
    """
    Multiply two tensors where last dimension is 2 and treat it as convex multiplication
    Args:
        a: [..., 2]
        b: [..., 2]
    """
    output = torch.zeros_like(a)
    output[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    output[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return output



def coords_to_sinusoidals_zero_centered(coords, num_sinusoidals=16, multiplier=2, freq_type='mult'):
    if freq_type not in ['mult', 'add']:
        raise RuntimeError("freq_type unknown. freq_type must be in ['mult', 'add']")
    
    sinusoidals = []
    for c in [1, 2, 4]:
        for i in range(num_sinusoidals):
            if freq_type == 'mult':
                sinusoidals.append(torch.cos((multiplier ** i) * np.pi * coords) * torch.exp(-c * torch.abs(coords)))
                sinusoidals.append(torch.sin((multiplier ** i) * np.pi *coords) * torch.exp(-c * torch.abs(coords)))
            elif freq_type == 'add':
                sinusoidals.append(torch.cos((multiplier * (i + 1)) * np.pi *coords) * torch.exp(-c * torch.abs(coords)))
                sinusoidals.append(torch.sin((multiplier * (i + 1)) * np.pi *coords) * torch.exp(-c * torch.abs(coords)))
            
    output = torch.cat(sinusoidals, dim=-1)
    return output


def coords_to_radials(coords, num_radials=32):
    assert num_radials > 5
    
    radials = []
    for i in range(num_radials):
        angle = torch.tensor((2 * np.pi * i) / num_radials)
        radials.append(torch.cos(angle) * coords[..., 0:1] + torch.sin(angle) * coords[..., 1:2])
    output = torch.cat(radials, dim=-1)
    return output

def coords_to_sincs(coords, num_sincs=16, multiplier=2):
    sincs = []
    for i in range(num_sincs):
        sincs.append(torch.sin((multiplier ** i) * coords) / ((multiplier ** i) * coords))
        sincs.append((1 - torch.cos((multiplier ** i) * coords)) / ((multiplier ** i) * coords))
    
    output = torch.cat(sincs, dim=-1)
    return output
    
def create_grid_sinusoidals(num_sinusoidals, H=64, W=64, multiplier=2):
    coords = create_linspace(-1, 1, H, W)
    return coords_to_sinusoidals(coords, num_sinusoidals, multiplier=multiplier)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
        
def plot_samples(samples, w, h, show=True):
    fig, axes = plt.subplots(w, h, figsize=(h * 3, w * 3))

    for i in range(h * w):
        axes[i // h, i % h].imshow(samples[i])

    if show:
        plt.show()
        
def sample_grid(get_samples, w, h, show=True):
    """
    Sampling from a model in a grid
    """
    output = get_samples(w * h)
    plot_samples(output, w, h, show=show)
    del output
    torch.cuda.empty_cache()

def num_trainable_params(module):
    return sum([p.numel() for p in module.parameters() if p.requires_grad])


class ResidualConnection(nn.Module):
    def __init__(self, *args, weight=1.0, normalize=False):
        super().__init__()

        self.weight = max(weight, 1e-6)
        self.normalize = normalize

        self.module = nn.Sequential(
            *args
        )

    def forward(self, x):
        output = self.module(x)

        output = output + self.weight * x
        if self.normalize:
            output = output / (1 + self.weight)

        return output
    
    
def modulated_linear(
    x,                          # Input tensor of shape [..., H, W, in_dim].
    weight,                     # Weight tensor of shape [out_dim, in_dim].
    # Modulation coefficients of shape [..., num_groups].
    styles,
    demodulate=True,           # Apply weight demodulation?
):
    assert x.shape[:-3] == styles.shape[:-1]
    assert x.shape[-1] == weight.shape[-1]
    H, W = x.shape[-3], x.shape[-2]
    shape_structure = x.shape[:-1]
    out_dim, in_dim = weight.shape

    num_groups = styles.shape[-1]
    elems_per_group = in_dim // num_groups

    x = x.flatten(end_dim=-2)  # [N * H * W, in_dim]
    styles = styles.flatten(end_dim=-2)  # [N, num_groups]
    N = styles.shape[0]

    # Splitting weight into groups
    # [out_dim, num_groups, elems_per_group]
    w = weight.view(out_dim, num_groups, elems_per_group)
    # Modulating
    w = w.unsqueeze(0)  # [1, out_dim, num_groups, elems_per_group]
    # [N, out_dim, num_groups, elems_per_group]
    w = w * styles.view(N, 1, num_groups, 1)
    # Demodulating
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3]) + 1e-8).rsqrt()  # [N, out_dim]
        w = w * dcoefs.view(N, out_dim, 1, 1)

    # Reshaping back to usual
    w = w.view(N, out_dim, in_dim)  # [N, out_dim, in_dim]

    # Applying linear transformation
    # [N * H * W, out_dim, in_dim]
    w = w[:, None, None, :, :].expand(-1, H, W, -1, -1).flatten(end_dim=-3)
    x = torch.bmm(w, x[..., None])[..., 0]  # [N, out_dim]

    x = x.view(*shape_structure, out_dim)  # [..., out_dim]
    return x


class StyleLinear(nn.Module):
    def __init__(self, input_dim, output_dim, w_dim, num_groups, activation=None, bias_init=0.0, demodulate=True, residual=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.demodulate = demodulate

        assert input_dim % num_groups == 0
        self.num_groups = num_groups

        self.weight = torch.nn.Parameter(torch.randn([output_dim, input_dim]))
        self.bias = torch.nn.Parameter(
            torch.full([output_dim], np.float32(bias_init)))

        self.affine = nn.Linear(w_dim, num_groups)
        self.activation = activation

        self.residual = residual if (
            self.input_dim == self.output_dim) else False

    def forward(self, x, w):
        """
        Args:
            x: Input tensor of shape [..., H, W, in_dim]
            w: Input tensor of shape [..., w_dim]
        Output:
            Tensor of shape [..., H, W, out_dim] by applying a style-modulated linear layer
        """
        styles = self.affine(w)  # [..., num_groups]

        output = x
        output = modulated_linear(
            output, self.weight, styles, demodulate=self.demodulate)  # [..., H, W, out_dim]
        output = output + \
            self.bias.view(*[1 for _ in range(output.dim() - 1)], -1)

        if self.activation is not None:
            output = self.activation(output)

        if self.residual:
            output = output + x

        return output
    
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)
            
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn
def compute_entropy(x, eps=1e-6):
    """
    Args:
        x: FloatTensor of shape [..., N] with non-negative values. 
    Output:
        output: FloatTensor of shape [...] 
    """
    if x.min() < 0.0:
        warnings.warn("Warning: Negative values found in compute_entropy(). Clamping to 0")
    
    x = F.relu(x) # Get rid of non-negatives
    x = x / (x.sum(dim=-1, keepdim=True) + eps)
    output = (-x * torch.log(x + eps)).sum(dim=-1)
    return output

class ScaleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        output = x * ((x ** 2).sum(dim=self.dim, keepdim=True) + 1e-6).rsqrt()
        return output
    
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
