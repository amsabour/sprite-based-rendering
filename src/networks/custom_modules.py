import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.profiler import record_function

import numpy as np
from einops import rearrange

import warnings
import math
import time

from utils import coords_to_sinusoidals, multiply_as_convex, create_linspace, linear_scale
from .stylegan import EqualConv2d, EqualLinear, ConstantInput, normalize_2nd_moment, Upsample
from .spatialstylegan import SpatialToRGB, SpatialStyledConv
from .op import fused_leaky_relu


@torch.jit.script
def custom_activation(x):
    x = F.relu(x)
    x = x / (x + 1.)
    return x

@torch.jit.script
def custom_activation2(x):
    return torch.tanh(torch.relu(x))

def custom_composite(shapes, depths, temperature=1.0):
    """
    Args:
        shapes: FloatTensor with shape [B, H, W, num_shapes]
        depths: FloatTensor with shape [B, num_shapes]
        temperature: Temperature parameter that determines strictness of depths
    Output:
        output: FloatTensor with shape [B, H, W, num_shapes] where:
            1) 0 <= output[...] <= 1
            2) sum(output[..., 0:num_shapes]) <= 1
            3) output[..., i] <= shapes[..., i]
    """
    # Naive implementation
    depths = depths[..., None, None, :].expand(*shapes.shape) # [B, H, W, num_shapes]
    depths = depths / max(temperature, 1e-6)

    if torch.max(shapes) > 1.0:
        warnings.warn(f'Warning: Values greater than 1 found in custom_composite(). Max value: {torch.max(shapes)}')
    if torch.min(shapes) < 0.0:
        warnings.warn(f'Warning: Values less than 0 found in custom_composite(). Min value: {torch.min(shapes)}')

    shapes = torch.clamp(shapes, 1e-6, 1 - 1e-6)

    pairwise_depths = depths.unsqueeze(-2) - depths.unsqueeze(-1)
    unnormalized_masks = torch.log(shapes) + (custom_activation2(pairwise_depths) * torch.log(1 - shapes).unsqueeze(-2)).sum(dim=-1)
    masks = F.softmax(unnormalized_masks, dim=-1)
    
    # num_shapes = shapes.shape[-1]
    # unnormalized_masks = []
    # for i in range(num_shapes):
    #     log_unnormalized_mask = torch.log(shapes[..., i:i+1])
    #     log_unnormalized_mask += (torch.log(1 - shapes) * custom_activation2(depths - depths[..., i].unsqueeze(-1))).sum(dim=-1, keepdim=True) 

    #     # for j in range(num_shapes):
    #     #     log_unnormalized_mask = log_unnormalized_mask + (torch.log((1 - shapes[..., j:j+1])) *
    #     #                                                      custom_activation2(depths[..., j:j+1] - depths[..., i:i+1]))
    #     # unnormalized_masks.append(torch.exp(log_unnormalized_mask))

    # unnormalized_masks = torch.cat(unnormalized_masks, dim=-1)
    # masks = unnormalized_masks / (torch.sum(unnormalized_masks, dim=-1, keepdim=True) + 1e-6)
    return masks

def reparametrize_trick(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = eps * std + mean
    return z

@torch.jit.script
def ease_function(x, min_threshold: float, max_threshold: float):
    x = x.double()
    x = (x - min_threshold) / (max_threshold - min_threshold)
    return torch.where(
        x < 0, 0.0, 
        torch.where(
            x > 1, 1.0, 
            -(torch.cos(np.pi * x) - 1.0) / 2.0
        )
    )


######## Planners ##############

class ActionDistribution:
    def __init__(self, params):
        super().__init__()
        self.mean, self.logvar = params.chunk(2, dim=-1)
        self.params = params

    def sample_action(self):
        sampled_action = reparametrize_trick(self.mean, self.logvar)
        return sampled_action

    def get_actions(self):
        actions = [self.sample_action() for _ in range(3)]
        return {'actions': actions}

class PredictionNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, cond_dim, hidden_dim=32, num_gaussians=3):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        self.input_dim = self.state_dim + self.cond_dim

        self.policy_module = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 2 * self.action_dim)
        )

    def forward(self, z_i, c):
        x = torch.cat([z_i, c], dim=-1)
        value = self.value_module(x)
        action_dist_params = self.policy_module(x)
        action_dist = ActionDistribution(action_dist_params)
        return {'value': value, 'action_dist_params': action_dist_params, 'action_dist': action_dist}

class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.input_dim = self.state_dim + self.action_dim
        self.output_dim = self.state_dim

        self.module = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        next_state = self.module(x)
        return {'next_state': next_state}

class DynamicsNetworkRNN(nn.Module):
    def __init__(self, per_layer_state_dim, action_dim, num_layers=3):
        super().__init__()
        
        self.per_layer_state_dim = per_layer_state_dim
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.state_dim = 2 * self.per_layer_state_dim * num_layers
        
        self.rnn_cells = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = self.action_dim if i == 0 else self.per_layer_state_dim
            self.rnn_cells.append(nn.LSTMCell(input_size, self.per_layer_state_dim))
        
    def forward(self, z, a):
        assert z.shape[-1] == self.state_dim
        
        next_zs = []
        next_z = None
        for i, rnn_cell in enumerate(self.rnn_cells):
            rnn_input = a if i == 0 else F.layer_norm(next_z[0], [self.per_layer_state_dim])
            next_z = rnn_cell(
                rnn_input, 
                (z[..., i * self.per_layer_state_dim * 2: i * self.per_layer_state_dim * 2 + self.per_layer_state_dim],
                 z[..., i * self.per_layer_state_dim * 2 + self.per_layer_state_dim: (i + 1) * self.per_layer_state_dim * 2])
            )
            next_zs.append(next_z[0])
            next_zs.append(next_z[1])
        
        next_state = torch.cat(next_zs, dim=-1)
        return {'next_state': next_state}

class PlannerMLP(nn.Module):
    def __init__(
        self, cond_dim, bg_dim, action_dim, num_actions,
        hidden_dim=256, lr_mul=0.01, **kwargs
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.bg_dim = bg_dim

        self.output_dim = self.num_actions * self.action_dim + self.bg_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            EqualLinear(self.cond_dim, self.hidden_dim,   lr_mul=lr_mul, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=lr_mul, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=lr_mul, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=lr_mul, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=lr_mul, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=lr_mul, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=lr_mul, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.output_dim, lr_mul=lr_mul, activation=None),
        )


    def forward(self, c):
        # Normalize input
        c = normalize_2nd_moment(c, dim=-1)

        # Main layers
        output = self.mlp(c.flatten(end_dim=-2)).view(*c.shape[:-1], -1)

        # Decompose output
        bg_feat, actions = output.split([self.bg_dim, self.action_dim * self.num_actions], dim=-1)
        actions = rearrange(actions, 'b (n d) -> b n d', n=self.num_actions, d=self.action_dim)
        return bg_feat, actions

class PlannerRecurrent1(nn.Module):
    def __init__(
        self, cond_dim, bg_dim, action_dim, num_actions,
        state_dim=32, hidden_dim=256,
        **kwargs
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.bg_dim = bg_dim

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.bg = nn.Linear(self.cond_dim, self.bg_dim)
        self.recurrent = nn.Sequential(
            EqualLinear(self.cond_dim + self.state_dim, self.hidden_dim, lr_mul=0.01, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=0.01, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=0.01, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.action_dim + self.state_dim, lr_mul=0.01, activation=None),
        )


    def forward(self, c):
        bg_feat = self.bg(c)
        
        z = torch.zeros(*c.shape[:-1], self.state_dim).to(c.device)
        actions = []
        for i in range(self.num_actions):
            action, z = self.recurrent(torch.cat([c, z]), dim=-1).split([self.action_dim, self.state_dim], dim=-1)
            actions.append(action)
        
        return bg_feat, actions

class PlannerRecurrent2(nn.Module):
    def __init__(
        self, cond_dim, bg_dim, action_dim, num_actions,
        state_dim=32, hidden_dim=128,
        **kwargs
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.bg_dim = bg_dim

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.bg = nn.Linear(self.cond_dim, self.bg_dim)
        self.policy = PredictionNetwork(self.state_dim, self.action_dim, self.cond_dim, hidden_dim=self.hidden_dim)
        self.dynamics = DynamicsNetwork(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim)


    def forward(self, c):
        bg_feat = self.bg(c)
        
        z = torch.zeros(*c.shape[:-1], self.state_dim).to(c.device)
        actions = []
        for i in range(self.num_actions):
            action = self.policy(z, c)['action_dist'].sample_action()
            actions.append(action)
            z = self.dynamics(z, action)['next_state']

        return bg_feat, actions 

class PlannerRecurrent3(nn.Module):
    def __init__(
        self, cond_dim, bg_dim, action_dim, num_actions,
        dynamics_hidden_dim=32, dynamics_num_layers=3, hidden_dim=128,
        **kwargs
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.bg_dim = bg_dim
        
        self.state_dim = dynamics_hidden_dim * dynamics_num_layers
        self.hidden_dim = hidden_dim

        self.bg = nn.Linear(self.cond_dim, self.bg_dim)
        self.policy = PredictionNetwork(self.state_dim, self.action_dim, self.cond_dim, hidden_dim=self.hidden_dim)
        self.dynamics = DynamicsNetworkRNN(dynamics_hidden_dim, self.action_dim, num_layers=dynamics_num_layers)

    def forward(self, c):
        bg_feat = self.bg(c)
        
        z = torch.zeros(*c.shape[:-1], self.state_dim).to(c.device)
        actions = []
        for i in range(self.num_actions):
            action = self.policy(z, c)['action_dist'].sample_action()
            actions.append(action)
            z = self.dynamics(z, action)['next_state']

        return bg_feat, actions

##################################

######## Sprite Assembler ########

class CoordinateDecoder(nn.Module):
    def __init__(self, cond_dim, out_dim, final_activation=None,
                 num_coords=2, num_sinusoidals=8, progressive=False,
                 radius_masking=True, hidden_dim=128, num_layers=4,
                 lr_mul=1):
        super().__init__()

        self.cond_dim = cond_dim
        self.hidden_dim=hidden_dim
        self.out_dim = out_dim
        self.num_layers = max(num_layers, 2)
        self.final_activation = final_activation

        self.num_coords = num_coords
        self.num_sinusoidals = num_sinusoidals
        self.progressive = progressive
        # 0 -> Everything locked, 1 -> Everything unlocked
        self.register_buffer("freq_unlock", torch.zeros(1))

        self.radius_masking = radius_masking
        self.lr_mul = lr_mul

        self.input_dim = 2 * self.num_coords * self.num_sinusoidals + self.cond_dim
        self.module = nn.ModuleList(
            [EqualLinear(self.input_dim,                  self.hidden_dim, activation='fused_lrelu', lr_mul=self.lr_mul)] + 
            [EqualLinear(self.hidden_dim + self.cond_dim, self.hidden_dim, activation='fused_lrelu', lr_mul=self.lr_mul) for _ in range(self.num_layers - 2)] +
            [EqualLinear(self.hidden_dim + self.cond_dim, self.out_dim,    activation='fused_lrelu', lr_mul=self.lr_mul)]
        )
        
        self.limits = (torch.arange(1, self.num_sinusoidals + 1) / (self.num_sinusoidals + 1)) # [1/(n+1), 2/(n+1), ..., n/(n+1)]

    def step(self, increment):
        self.freq_unlock += increment

    def forward(self, coords, c):
        # Embed the coords
        embedded_coords = coords_to_sinusoidals(
            coords / 10, num_sinusoidals=self.num_sinusoidals, multiplier=2, freq_type='mult'
        )

        # Progressive frequency encoding
        if self.progressive:
            self.limits = self.limits.to(coords.device)
            limits = self.limits  # [1/(n+1), 2/(n+1), ..., n/(n+1)]
            weights = torch.clamp((self.freq_unlock - limits) * (self.num_sinusoidals + 1), 0.0, 1.0)
            expanded_weights = weights[:, None].expand(-1, 4).flatten()
            embedded_coords = embedded_coords * expanded_weights[(None,) * (coords.dim() - 1)]

        # Passing through the module
        c = c[..., None, None, :].expand(*((-1,) * (c.dim() - 1)), embedded_coords.shape[-3], embedded_coords.shape[-2], -1)
        
        output_shape = embedded_coords.shape[:-1]
        output = embedded_coords.flatten(end_dim=-2)
        c = c.flatten(end_dim=-2)
        for i, layer in enumerate(self.module):
            output = torch.cat([output, c], dim=-1)
            output = layer(output)
        output = output.view(*output_shape, -1)
        
        if self.final_activation:
            output = self.final_activation(output)

        # Apply out-of-bounds masking
        if self.radius_masking:
            radius = ((coords ** 2).sum(dim=-1, keepdim=True)).sqrt()
            bounds_limiter = 1 - torch.tanh(F.relu(radius - 1.0))
            output = output * bounds_limiter

        return output

class SpritesAndTexturesCoordinateDecoder(nn.Module):
    def __init__(self, shape_dim, texture_dim, shape_out_dim, texture_out_dim, shape_activation=None, texture_activation=None,
                 num_coords=2, num_sinusoidals=8, progressive=False,
                 shared_num_layers=2, additional_num_layers=1,
                 hidden_dim=128, lr_mul=1):
        super().__init__()

        self.shape_dim = shape_dim
        self.shape_out_dim = shape_out_dim
        self.shape_activation = shape_activation
        self.texture_dim = texture_dim
        self.texture_out_dim = texture_out_dim
        self.texture_activation = texture_activation
        self.hidden_dim=hidden_dim
        
        self.shared_num_layers = shared_num_layers
        self.additional_num_layers = additional_num_layers

        self.num_coords = num_coords
        self.num_sinusoidals = num_sinusoidals
        self.progressive = progressive
        # 0 -> Everything locked, 1 -> Everything unlocked
        self.register_buffer("freq_unlock", torch.zeros(1))

        self.lr_mul = lr_mul

        self.input_dim = 2 * self.num_coords * self.num_sinusoidals + self.shape_dim

        self.shared_modules = nn.Sequential(*(
            [EqualLinear(self.input_dim,  self.hidden_dim, activation='fused_lrelu', lr_mul=self.lr_mul)] + 
            [EqualLinear(self.hidden_dim, self.hidden_dim, activation='fused_lrelu', lr_mul=self.lr_mul) for _ in range(self.shared_num_layers)]
        ))

        self.to_shape = EqualLinear(self.hidden_dim, self.shape_out_dim, activation=None, lr_mul=self.lr_mul)

        self.to_textures = nn.Sequential(*(
            [EqualLinear(self.hidden_dim + self.texture_dim,  self.hidden_dim, activation='fused_lrelu', lr_mul=self.lr_mul)] + 
            [EqualLinear(self.hidden_dim, self.hidden_dim, activation='fused_lrelu', lr_mul=self.lr_mul) for _ in range(self.additional_num_layers)] + 
            [EqualLinear(self.hidden_dim, self.texture_out_dim, activation=None, lr_mul=self.lr_mul)]
        ))

        self.limits = (torch.arange(1, self.num_sinusoidals + 1) / (self.num_sinusoidals + 1)) # [1/(n+1), 2/(n+1), ..., n/(n+1)]

    def step(self, increment):
        self.freq_unlock += increment

    def forward(self, coords, shape_cond, texture_cond):
        # Embed the coords
        embedded_coords = coords_to_sinusoidals(
            coords / 10, num_sinusoidals=self.num_sinusoidals, multiplier=2, freq_type='mult'
        )

        # Progressive frequency encoding
        if self.progressive:
            self.limits = self.limits.to(coords.device)
            limits = self.limits  # [1/(n+1), 2/(n+1), ..., n/(n+1)]
            weights = torch.clamp((self.freq_unlock - limits) * (self.num_sinusoidals + 1), 0.0, 1.0)
            expanded_weights = weights[:, None].expand(-1, 4).flatten()
            embedded_coords = embedded_coords * expanded_weights[(None,) * (coords.dim() - 1)]

        # Passing through the module
        shape_cond = shape_cond[..., None, None, :].expand(*((-1,) * (shape_cond.dim() - 1)), embedded_coords.shape[-3], embedded_coords.shape[-2], -1)
        texture_cond = texture_cond[..., None, None, :].expand(*((-1,) * (texture_cond.dim() - 1)), embedded_coords.shape[-3], embedded_coords.shape[-2], -1)
        
        output_shape = embedded_coords.shape[:-1]
        shape_cond = shape_cond.flatten(end_dim=-2)
        texture_cond = texture_cond.flatten(end_dim=-2)

        shared_features = self.shared_modules(torch.cat([embedded_coords.flatten(end_dim=-2), shape_cond.flatten(end_dim=-2)], dim=-1))

        shape_output = self.to_shape(shared_features)
        shape_output = shape_output.view(*output_shape, -1)
        texture_output = self.to_textures(torch.cat([shared_features, texture_cond], dim=-1))
        texture_output = texture_output.view(*output_shape, -1)

        if self.shape_activation:
            shape_output = self.shape_activation(shape_output)
        if self.texture_activation:
            texture_output = self.texture_activation(texture_output)        

        # Apply out-of-bounds masking for shapes
        if True:
            radius = ((coords ** 2).sum(dim=-1, keepdim=True)).sqrt()
            bounds_limiter = 1 - torch.tanh(F.relu(radius - 1.0))
            shape_output = shape_output * bounds_limiter

        return shape_output, texture_output

class StarCoordinateDecoder(nn.Module):
    def __init__(self, 
        cond_dim, out_dim,
        num_coords=2, num_sinusoidals=8,
        min_radius=0.3, max_radius=1, 
        progressive=True,
    ):
        super().__init__()

        self.cond_dim = cond_dim

        self.out_dim = out_dim

        self.num_coords = num_coords
        self.num_sinusoidals = num_sinusoidals
        # 0 -> Everything locked, 1 -> Everything unlocked
        self.register_buffer("freq_unlock", torch.zeros(1))

        self.min_r = min_radius
        self.max_r = max_radius

        if self.num_sinusoidals > 0:
            self.input_dim = 2 * (self.num_coords - 1) * self.num_sinusoidals + self.cond_dim
        else:
            self.input_dim = 2 * (self.num_coords - 1) + self.cond_dim
        
        self.module = nn.ModuleList([
            EqualLinear(self.input_dim, 64, activation='fused_lrelu'),
            EqualLinear(64 + self.cond_dim, 64, activation='fused_lrelu'),
            EqualLinear(64 + self.cond_dim, 64, activation='fused_lrelu'),
            EqualLinear(64 + self.cond_dim, self.out_dim, activation=None),
        ])

        self.progressive = progressive

    def step(self, increment):
        self.freq_unlock += increment
        self.freq_unlock = torch.clamp(self.freq_unlock, 0, 1)
    
    def forward(self, coords, c):
        original_c = c

        # Normalize coords to have length of sqrt(d)
        normalized_coords = normalize_2nd_moment(coords, dim=-1)
        thetas = torch.cat([
            torch.atan2(normalized_coords[..., i:i+1], normalized_coords[..., -1:])
            for i in range(self.num_coords - 1)
        ], dim=-1) / np.pi
        
        # Apply positional embeddings to the thetas
        if self.num_sinusoidals > 0:
            embedded_thetas = coords_to_sinusoidals(
                thetas, num_sinusoidals=self.num_sinusoidals, multiplier=2, freq_type='mult'
            )

            # Progressive frequency encoding
            if self.progressive:
                limits = (torch.arange(0, self.num_sinusoidals).to(coords.device) / (self.num_sinusoidals))  # [0/n, 1/n, ..., (n-1)/n]
                weights = torch.clamp((self.freq_unlock - limits) * (self.num_sinusoidals), 0.0, 1.0)
                expanded_weights = weights[:, None].expand(-1, 2 * (self.num_coords - 1)).flatten()
                embedded_thetas = embedded_thetas * expanded_weights[(None,) * (coords.dim() - 1)]
        else:
            embedded_thetas = coords_to_sinusoidals(
                thetas, num_sinusoidals=1, multiplier=2, freq_type='mult'
            )

        # Passing through the module
        c = c[..., None, None, :].expand(*((-1,) * (c.dim() - 1)), embedded_thetas.shape[-3], embedded_thetas.shape[-2], -1)
        output_shape = embedded_thetas.shape[:-1]
        output = embedded_thetas.flatten(end_dim=-2)
        c = c.flatten(end_dim=-2)
        
        for i, layer in enumerate(self.module):
            output = torch.cat([output, c], dim=-1)
            output = layer(output)
        output = output.view(*output_shape, -1)
        dist_to_border = torch.clamp(output, self.min_r, self.max_r) # [...., out_dim]
        
        # Compute the radius of the coords
        radius = coords.pow(2).sum(dim=-1, keepdim=True).sqrt()
        output = ease_function(dist_to_border - radius, -0.5, 0).float()

        try:
            theta_grads = torch.autograd.grad(outputs=dist_to_border.sum(), inputs=thetas, create_graph=True, only_inputs=True)[0]
            cond_grads = torch.autograd.grad(outputs=dist_to_border.mean((-2, -3)).sum(), inputs=original_c, create_graph=True, only_inputs=True)[0]
        except RuntimeError:
            theta_grads = None 
            cond_grads = None

        return output, (theta_grads, cond_grads)

class EqualLinear2(nn.Module):
    """
    The only difference between this and EqualLinear is the initialization. 
    """
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1, 1).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = lr_mul / math.sqrt(in_dim)
        self.lr_mul = lr_mul

    def forward(self, input):
        assert input.shape[-1] == self.weight.shape[-1], \
            f'Input shape {input.shape[-1]} != weight shape {self.weight.shape[-1]}'

        if self.activation:
            out = F.linear(input, self.weight * self.scale) 
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}, {'' if self.bias is not None else 'no '}bias, lr_mul={self.lr_mul})"
        )

class ICouplingBlock(nn.Module):
    def __init__(
        self, cond_dim, num_coords=2,
        hidden_dim=128, num_sinusoidals=2, proj_dim=128, lr_mul=1
    ):
        super().__init__()
        
        self.cond_dim = cond_dim
        self.num_coords = num_coords
        self.hidden_dim = hidden_dim
        
        self.num_sinusoidals = num_sinusoidals
        self.proj_dim = proj_dim
        self.coord_input_dim = self.proj_dim if self.num_sinusoidals == 0 else 2 * self.num_sinusoidals
        
        self.lr_mul = lr_mul

        self.x_proj = nn.Sequential(
            EqualLinear(1, 2 * self.proj_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            EqualLinear(2 * self.proj_dim, self.proj_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
        )

        self.x_module = nn.ModuleList([
            EqualLinear2(self.coord_input_dim + self.cond_dim, self.hidden_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            # EqualLinear2(self.hidden_dim + self.cond_dim, self.hidden_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            # EqualLinear2(self.hidden_dim + self.cond_dim, self.hidden_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            EqualLinear2(self.hidden_dim + self.cond_dim, 2, lr_mul=self.lr_mul),
        ])
        
        self.y_proj = nn.Sequential(
            EqualLinear(1, 2 * self.proj_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            EqualLinear(2 * self.proj_dim, self.proj_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
        )
        self.y_module = nn.ModuleList([
            EqualLinear2(self.coord_input_dim + self.cond_dim, self.hidden_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            # EqualLinear2(self.hidden_dim + self.cond_dim, self.hidden_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            # EqualLinear2(self.hidden_dim + self.cond_dim, self.hidden_dim, lr_mul=self.lr_mul, activation='fused_lrelu'),
            EqualLinear2(self.hidden_dim + self.cond_dim, 2, lr_mul=self.lr_mul),
        ])
        
        self.hardtanh = nn.Hardtanh(min_val=-5, max_val=5)

        # Initialize to identity
        self.x_module[-1].weight.data.fill_(0)
        self.y_module[-1].weight.data.fill_(0)

    def forward(self, coords, c):
        initial_shape = coords.shape[:-1]
        coords = coords.flatten(end_dim=-2)
        c = c.flatten(end_dim=-2)

        x = coords[..., 0:1]
        y = coords[..., 1:2]
        
        # Apply x
        x_input = self.x_proj(x) if self.num_sinusoidals == 0 else coords_to_sinusoidals(x / 10, num_sinusoidals=self.num_sinusoidals, multiplier=2, freq_type='mult')
        for layer in self.x_module:
            x_input = layer(torch.cat([x_input, c], dim=-1))
        y_log_scale, y_bias = x_input.chunk(2, dim=-1)
        y_log_scale = self.hardtanh(y_log_scale)
        y = y * torch.exp(y_log_scale) + y_bias

        # Apply y
        y_input = self.y_proj(y) if self.num_sinusoidals == 0 else coords_to_sinusoidals(y / 10, num_sinusoidals=self.num_sinusoidals, multiplier=2, freq_type='mult')
        for layer in self.y_module:
            y_input = layer(torch.cat([y_input, c], dim=-1))
        x_log_scale, x_bias = y_input.chunk(2, dim=-1)
        x_log_scale = self.hardtanh(x_log_scale)
        x = x * torch.exp(x_log_scale) + x_bias

        warped_coords = torch.cat([x, y], dim=-1).view(*initial_shape, -1)
        log_jacobian = (x_log_scale + y_log_scale).view(*initial_shape, -1)
        return warped_coords, log_jacobian

class CoordinateDeformer(nn.Module):
    def __init__(self, 
        cond_dim, num_coords, deform_block, num_blocks=3
    ):
        super().__init__()

        self.cond_dim = cond_dim
        self.num_coords = num_coords
        self.num_blocks = num_blocks

        self.deformations = nn.ModuleList(
            [deform_block(self.cond_dim, self.num_coords) for _ in range(self.num_blocks)]
        )
    
    def step(self, increment):
        pass

    def forward(self, coords, c):
        c = normalize_2nd_moment(c, dim=-1)
        
        initial_coords = coords
        original_c = c
        
        # Passing through the module
        c = c[..., None, None, :].expand(*((-1,) * (c.dim() - 1)), coords.shape[-3], coords.shape[-2], -1)
        coords_shape = coords.shape[:-1] 
        coords = coords.flatten(end_dim=-2)
        c = c.flatten(end_dim=-2)
        
        log_jacobian = 0
        zero_coords = torch.zeros(*original_c.shape[:-1], 2, device=c.device).flatten(end_dim=-2)
        for _, layer in enumerate(self.deformations):
            coords, layer_log_jacobian = layer(coords, c)
            log_jacobian += layer_log_jacobian
            
            zero_coords, _ = layer(zero_coords, original_c)
        
        coords = coords.view(*coords_shape, -1)
        zero_coords = zero_coords.view(*original_c.shape[:-1], -1)
        coords = coords - zero_coords[..., None, None, :]

        # Compute the radius of the coords
        radius = coords.pow(2).sum(dim=-1, keepdim=True).sqrt()
        output = ease_function(1 - radius, -0.5, 0).float()
        
        output = output.view(*coords_shape, 1)
        log_jacobian = log_jacobian.view(*coords_shape, 1)
        

        return output, (log_jacobian, coords, initial_coords)

class SpriteAssembler(nn.Module):
    def __init__(
        self, 
        shape_style_dim=32, shape_encoding_dim=8,
        shape_progressive=False, shape_num_sinusoidals=8, 
        use_textures=True, texture_progressive=True, texture_num_sinusoidals=8,
        shape_library_lr_mul=1.00, texture_library_lr_mul=0.01,
    ):
        super().__init__()

        self.shape_style_dim = shape_style_dim
        self.shape_encoding_dim = shape_encoding_dim
        self.shape_library_lr_mul = shape_library_lr_mul
        self.texture_library_lr_mul = texture_library_lr_mul
        
        # Action parts and postprocesses
        self.action_configs = [
            ('style', self.shape_style_dim, None), 
            ('shape', self.shape_encoding_dim, None), 
            
            ('alpha', 1, lambda x: torch.ones_like(x)),
            ('depth', 1, None),
            ('translation', 2, lambda x: torch.tanh(x) * 0.9 + (torch.rand_like(x) * 0.1 - 0.05)), # Perturb with a uniform noise between [-0.05, 0.05]
            ('rotation', 1, lambda x: torch.zeros_like(x)),
            ('scale', 1, lambda x: linear_scale(torch.tanh(x), new_min=0.1, new_max=0.5) + (torch.rand_like(x) * 0.1 - 0.05)), # Between [0.2, 0.8] with noise between [-0.05, 0.05]
        ]
        self.action_configs = [(x[0], x[1], x[2] if x[2] is not None else (lambda x: x)) for x in self.action_configs]
        self.action_dim = sum([x[1] for x in self.action_configs])

        # Sprite library
        self.shape_progressive = shape_progressive
        self.shape_num_sinusoidals = shape_num_sinusoidals
        # self.shape_library = StarCoordinateDecoder(
        #     cond_dim=self.shape_encoding_dim, out_dim=1, num_coords=2, 
        #     num_sinusoidals=self.shape_num_sinusoidals, progressive=self.shape_progressive,
        #     min_radius=0.3, max_radius=1.2, 
        # )
        # self.shape_library = CoordinateDeformer(
        #     cond_dim=self.shape_encoding_dim, num_coords=2, 
        #     deform_block=lambda x, y: ICouplingBlock(x, y, num_sinusoidals=self.shape_num_sinusoidals, lr_mul=self.shape_library_lr_mul),
        #     num_blocks=6
        # )
        self.shape_library = CoordinateDecoder(
            cond_dim=self.shape_encoding_dim, out_dim=1, final_activation=nn.Sigmoid(),
            num_coords=2, num_sinusoidals=self.shape_num_sinusoidals, progressive=self.shape_progressive, 
            radius_masking=True, num_layers=4
        )
        
        # Texture library
        self.texture_progressive = texture_progressive
        self.texture_num_sinusoidals = texture_num_sinusoidals
        self.use_textures = use_textures
        self.texture_library = CoordinateDecoder(
            cond_dim=self.shape_style_dim, 
            # out_dim=self.shape_style_dim, 
            out_dim=3, 
            final_activation=None,
            num_coords=2, num_sinusoidals=self.texture_num_sinusoidals, progressive=self.texture_progressive, 
            radius_masking=False,
            lr_mul=self.texture_library_lr_mul,
            num_layers=4,
        )

        self.sprite_and_texture_library = SpritesAndTexturesCoordinateDecoder(
            shape_dim=self.shape_encoding_dim,
            texture_dim=self.shape_style_dim,
            shape_out_dim=1,
            texture_out_dim=3,
            shape_activation=nn.Sigmoid(),
            texture_activation=None,
            num_coords=2, num_sinusoidals=self.shape_num_sinusoidals,
            progressive=self.shape_progressive,
            lr_mul=self.texture_library_lr_mul,
        )
        
        self.compositing_temperature = 1.0

        self.coords = None

    def postprocess_action(self, action):
        assert action.shape[-1] == self.action_dim
        action_partitioned = torch.split(action, [x[1] for x in self.action_configs], dim=-1)
        action_processed = [x[2](action_partitioned[i]) for i, x in enumerate(self.action_configs)]
        return torch.cat(action_processed, dim=-1)
    
    def composite(self, sprites, depths):
        sprites = torch.cat([sprites, torch.ones_like(sprites[..., 0:1])], dim=-1)
        depths = torch.cat([depths / self.compositing_temperature, torch.ones_like(depths[..., 0:1]) * -100], dim=-1)
        with record_function("custom_composite"):
            masks = custom_composite(sprites, depths, temperature=1.0)
        return masks

    def compute_shapes(self, actions, coords):
        all_shapes_encodings = actions['shape']
        all_shape_depths = actions['depth']
        all_shape_translations = actions['translation']
        all_shape_scales = actions['scale']
        all_shape_rotations = actions['rotation']
        all_shape_alphas = actions['alpha']
        all_shape_styles = actions['style']
        
        batch_size = all_shapes_encodings.shape[0]
        num_shapes = all_shapes_encodings.shape[1]
        
        # Transfer base coordinates to new space
        shape_aligned_coords = coords[:, None, :, :, :].expand(-1, num_shapes, -1, -1, -1) # [batch_size, num_actions, H, W, 2]
        # Translation
        shape_aligned_coords = shape_aligned_coords - all_shape_translations[:, :, None, None, :] # [batch_size, num_actions, H, W, 2]
        # Rotation + Scale
        shape_rs = torch.cat([torch.cos(all_shape_rotations) / all_shape_scales, torch.sin(all_shape_rotations) / all_shape_scales], dim=-1)
        shape_aligned_coords = multiply_as_convex(shape_aligned_coords, shape_rs[:, :, None, None, :]) # rotate, scale     

        # # Compute shape sprite
        # sprites, sprite_extras = self.get_sprite(shape_aligned_coords, all_shapes_encodings)
        # sprites = sprites * all_shape_alphas[..., None, None, :] # [batch_size, num_actions, H, W, 1]
        # sprites = sprites[..., 0].permute(0, 2, 3, 1) # [B, H, W, num_actions]
        
        # # Compute shape texture
        # textures = self.get_texture(shape_aligned_coords, all_shape_styles).permute(0, 2, 3, 1, 4) # [B, H, W, num_shapes, shape_style_dim]
        simple_textures = self.get_texture(shape_aligned_coords, all_shape_styles, use_textures=False).permute(0, 2, 3, 1, 4) # [B, H, W, num_shapes, shape_style_dim]

        # Compute sprites and textures 
        sprites, textures = self.get_sprite_and_texture(shape_aligned_coords, all_shapes_encodings, all_shape_styles)
        sprites = sprites * all_shape_alphas[..., None, None, :] # [batch_size, num_actions, H, W, 1]
        sprites = sprites[..., 0].permute(0, 2, 3, 1) # [B, H, W, num_actions]
        textures = textures.permute(0, 2, 3, 1, 4) # [B, H, W, num_shapes, shape_style_dim]
        
        # Compute masks
        masks = self.composite(sprites, all_shape_depths[..., 0])

        shape_aligned_coords = shape_aligned_coords.permute(0, 2, 3, 1, 4) # [B, H, W, num_shapes, 2]
        
        # return masks, (sprites, sprite_extras), (textures, simple_textures, shape_aligned_coords)
        return masks, (sprites, None), (textures, simple_textures, shape_aligned_coords)
        

    def get_sprite_and_texture(self, coords, shape_cond, texture_cond):
        sprites, textures = self.sprite_and_texture_library(coords, shape_cond, texture_cond)

        # ### Enforce non-empty sprite
        min_r = 0.3
        radius = ((coords ** 2).sum(dim=-1, keepdim=True)).sqrt()
        sprites = torch.clamp(sprites + torch.sigmoid((-radius + min_r) / min_r * 5), 0, 1.0)

        return sprites, textures


    def get_sprite(self, coords, cond):
        if isinstance(self.shape_library, CoordinateDeformer):
            sprite, extras = self.shape_library(coords, cond)
        else:
            sprite = self.shape_library(coords, cond)
            extras = None
        
        # ### Enforce non-empty sprite
        min_r = 0.3
        radius = ((coords ** 2).sum(dim=-1, keepdim=True)).sqrt()
        sprite = torch.clamp(sprite + torch.sigmoid((-radius + min_r) / min_r * 5), 0, 1.0)
        
        return sprite, extras

    def get_texture(self, coords, cond, use_textures=None):
        if use_textures is None:
            use_textures = self.use_textures

        if use_textures:
            texture = self.texture_library(coords, cond)
        else:
            texture = cond[..., None, None, :].expand(*(-1,) * (cond.dim() - 1), coords.shape[-3], coords.shape[-2], -1)
        
        # texture = normalize_2nd_moment(texture, dim=1)
        return texture

    def forward(self, bg_feat, actions, size=64):
        batch_size = bg_feat.shape[0]
        device = bg_feat.device
        
        with record_function("assembler_coord_gen"):
            # Calculate pixel coordinates
            if self.coords is None:
                self.coords = (create_linspace(H=size, W=size, as_center_points=True)).unsqueeze(0).to(device)
            coords = self.coords.to(device)
            coords = coords.repeat(batch_size, 1, 1, 1)
        
        with record_function("assembler_bg_textures"):
            # Background
            bg_texture  = self.get_texture(coords, bg_feat, use_textures=True)
            bg_simple_texture = self.get_texture(coords, bg_feat, use_textures=False)
            if actions.shape[1] == 0:
                return bg_texture, bg_simple_texture
        
        # Postprocess actions (This is moved to the main forward function to remove randomness between different render steps of a scene)
        # actions = [self.postprocess_action(x) for x in actions]
        
        with record_function("assembler_action_partitioning"):
            # Split actions into parts
            # actions_concatenated = torch.cat([x[:, None, :] for x in actions], dim=1) # [B, num_actions, action_dim]
            actions_partitioned = torch.split(actions, [x[1] for x in self.action_configs], dim=-1)
            actions = {x[0]: actions_partitioned[i] for i, x in enumerate(self.action_configs)}
        
        with record_function("assembler_compute_shapes"):
            # Generate and composite shapes
            masks, (sprites, sprite_extras), (textures, simple_textures, texture_coords) = self.compute_shapes(actions, coords)
        
        with record_function("assembler_combine_features"):
            # Combine features
            textures = torch.cat([textures, bg_texture.unsqueeze(-2)], dim=-2) # [batch_size, H, W, num_shapes+1, shape_style_dim]
            simple_textures = torch.cat([simple_textures, bg_simple_texture.unsqueeze(-2)], dim=-2) # [batch_size, H, W, num_shapes+1, shape_style_dim]
            textures = torch.clamp(textures, -1, 1) # In order to avoid too-big textures that get automatically "fixed" using small masks
            feature_map = (masks[..., None] * textures).sum(dim=-2) # [batch_size, H, W, shape_style_dim]

        return (
            feature_map, 
            {'masks': masks, 'sprites': sprites, 'textures': textures, 'actions': actions, 'simple_textures': simple_textures, 
            'sprite_extras': sprite_extras, 'texture_extras': torch.cat([texture_coords, coords.unsqueeze(-2)], dim=-2)}
        )

##################################

class NonstyledToRGB(nn.Module):
    def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1], c_out=3):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = EqualConv2d(in_channel, c_out, 1, bias=True)

    def forward(self, inputs, skip=None):
        out = self.conv(inputs)

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out

class PixelClassifier(nn.Module):
    # Taken from https://github.com/nv-tlabs/datasetGAN_release
    def __init__(self, num_classes, input_dim):
        super().__init__()
        
        # if num_classes < 32:
        #     self.layers = nn.Sequential(
        #         nn.Linear(input_dim, 128),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(num_features=128),
        #         nn.Linear(128, 32),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(num_features=32),
        #         nn.Linear(32, num_classes),
        #     )
        # else:
        #     self.layers = nn.Sequential(
        #         nn.Linear(input_dim, 256),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(num_features=256),
        #         nn.Linear(256, 128),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(num_features=128),
        #         nn.Linear(128, num_classes),
        #     )
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            nn.Softmax(),
        )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)



class StyleDecoder(nn.Module):
    def __init__(
        self,
        size: int = 256,
        style_dim: int = 512,
        channel_multiplier: int = 2,
        c_out: int = 3,
        c_model: int = 64,
        size_in: int = 16,
        const_in: bool = False,

        seg_classes: int = 9,
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        self.channel_multiplier = channel_multiplier
        self.c_out = c_out
        self.c_model = c_model
        self.size_in = size_in
        self.const_in = const_in
        self.seg_classes = seg_classes
        
        blur_kernel = [1, 3, 3, 1]

        self.channels = {
            4: self.c_model * 8,
            8: self.c_model * 8,
            16: self.c_model * 8,
            32: self.c_model * 8,
            64: self.c_model * 4 * self.channel_multiplier,
            128: self.c_model * 2 * self.channel_multiplier,
            256: self.c_model * self.channel_multiplier,
            512: self.c_model // 2 * self.channel_multiplier,
            1024: self.c_model // 4 * self.channel_multiplier,
        }

        self.num_total_channels = 0

        if self.const_in:
            self.input = ConstantInput(self.channels[self.size_in], size=size_in)
        
        input_dim = self.channels[self.size_in] if self.const_in else self.style_dim
        self.conv1 = SpatialStyledConv(
            input_dim, self.channels[self.size_in], 3, self.style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = SpatialToRGB(self.channels[self.size_in], self.style_dim, upsample=False, c_out=self.c_out)      
        self.num_total_channels += self.channels[self.size_in]
        self.to_seg1 = NonstyledToRGB(self.channels[self.size_in], c_out=self.seg_classes)
        
        self.log_size = int(math.log(self.size, 2))
        self.log_size_in = int(math.log(self.size_in, 2))
        self.c_in = self.channels[self.size_in]
        self.num_layers = (self.log_size - self.log_size_in) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_segs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[self.size_in]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 1) // 2 + self.log_size_in 
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(self.log_size_in + 1, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                SpatialStyledConv(
                    in_channel,
                    out_channel,
                    3,
                    self.style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            
            self.convs.append(
                SpatialStyledConv(
                    out_channel, out_channel, 3, self.style_dim, blur_kernel=blur_kernel
                )
            )
            self.num_total_channels += out_channel

            self.to_rgbs.append(SpatialToRGB(out_channel, self.style_dim, c_out=self.c_out))
            self.to_segs.append(NonstyledToRGB(out_channel, c_out=self.seg_classes))

            in_channel = out_channel

        

        self.n_latent = (self.log_size - self.log_size_in + 1) * 2

        self.to_seg = PixelClassifier(512, self.num_total_channels)


    def get_noise_like(self):
        return [torch.randn_like(getattr(self.noises, f"noise_{i}")) for i in range(self.num_layers)]
    
    def forward(
        self,
        styles,
        input=None,
        return_latents=False,
        return_image_only=True,
        noise=None,
        randomize_noise=True,
        return_features=False
    ):
        if noise == 'const':
            noise = [
                getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
            ]
        
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if isinstance(styles, dict):
            masks = styles['masks'] # [batch_size, H, W, num_shapes+1]
            masks = torch.cat([masks, torch.zeros(*masks.shape[:-1], self.seg_classes - masks.shape[-1]).to(masks.device)], dim=-1)
            styles['masks'] = masks

            textures = styles['textures'] # [batch_size, H, W, num_shapes+1, shape_style_dim]
            textures = torch.cat([textures, torch.zeros(*textures.shape[:3], self.seg_classes - textures.shape[-2], textures.shape[-1]).to(textures.device)], dim=-2)
            styles['textures'] = textures

        latent = [styles] * self.n_latent
        
        if self.const_in:
            if isinstance(styles, dict):
                out = self.input(latent[0]['textures'])
            else:
                out = self.input(latent[0])
        else:
            if isinstance(styles, dict):
                masks = styles['masks'] # [batch_size, H, W, num_shapes+1]
                masks = F.interpolate(
                    rearrange(masks, 'b H W s -> b s H W'), (self.size_in, self.size_in), mode='bilinear', align_corners=False#, antialias=True
                )

                textures = styles['textures'] # [batch_size, H, W, num_shapes+1, shape_style_dim]
                num_textures = textures.shape[-2]
                texture_dim = textures.shape[-1]
                textures = F.interpolate(rearrange(textures, 'b H W s d -> b (s d) H W'), (self.size_in, self.size_in), mode='bilinear')
                textures = rearrange(textures, 'b (s d) H W -> b d s H W', s=num_textures, d=texture_dim)
                
                out = (masks[:, None] * textures).sum(dim=2)
            else:
                out = F.interpolate(styles, (self.size_in, self.size_in), mode='bilinear', align_corners=False)#, antialias=True)
            
            # out = normalize_2nd_moment(out, dim=1)

        out = self.conv1(out, latent[0], noise=noise[0])

        self.outs = [out]
        skip = self.to_rgb1(out, latent[1])
        # skip_seg = self.to_seg1(out)
        reshaped_masks = rearrange(styles['masks'], 'b H W d -> b d H W')
        skip_seg = torch.log(F.interpolate(reshaped_masks, (self.size_in, self.size_in), mode='bilinear') + 1e-6)

        rgbs = [skip]
        masks = [styles['masks'], rearrange(F.softmax(skip_seg, dim=1), 'b d H W -> b H W d')]

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb, to_seg in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs, self.to_segs
        ):
            out = conv1(out, latent[i], noise=noise1)
            out = conv2(out, latent[i + 1], noise=noise2)
            self.outs.append(out)
            skip = to_rgb(out, latent[i + 2], skip)

            if skip.shape[-1] <= reshaped_masks.shape[-1]:
                skip_seg =  torch.log(F.interpolate(reshaped_masks, (skip.shape[-1], skip.shape[-1]), mode='bilinear') + 1e-6)
            else:
                try:
                    skip_seg = to_seg(out.detach(), skip_seg) # Detach masks to avoid influencing the features
                except RuntimeError as e:
                    print(f"Out shape:{out.shape}, Skip_seg shape: {skip_seg.shape}", flush=True)
                    raise e
            
            i += 2

            rgbs.append(skip)
            cur_res_masks =  rearrange(F.softmax(skip_seg, dim=1), 'b d H W -> b H W d')
            masks.append(cur_res_masks)
            if i + 2 < len(latent): # Use higher-res masks the further you go
                latent[i]['masks'] = cur_res_masks
                latent[i + 1]['masks'] = cur_res_masks
                latent[i + 2]['masks'] = cur_res_masks

        # image = torch.cat([skip, skip_seg], dim=1)
        image = (rgbs, masks)

        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        elif return_image_only:
            return image
        else:
            return image, None

class MultiShapeStyleLinearApproach(nn.Module):
    def __init__(
        self, hidden_dim=64, hidden_action_dim=32, planner_type='mlp', num_actions=8,

        shape_style_dim=32, shape_encoding_dim=8, 
        shape_progressive=False, shape_num_sinusoidals=8, 
        use_textures=True, texture_progressive=True, texture_num_sinusoidals=8,
        
        to_rgb_type='none', output_size=64, const_in=True, size_in=4, c_model=32,

        planner_lr_mul=0.01, shape_library_lr_mul=1.0, texture_library_lr_mul=0.01,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_action_dim = hidden_action_dim
        self.num_actions = num_actions
        self.shape_style_dim = shape_style_dim
        self.output_size = output_size

        self.assembler = SpriteAssembler(
            shape_style_dim=shape_style_dim, 
            shape_encoding_dim=shape_encoding_dim,
            shape_progressive=shape_progressive, shape_num_sinusoidals=shape_num_sinusoidals, 
            use_textures=use_textures, texture_progressive=texture_progressive, texture_num_sinusoidals=texture_num_sinusoidals,
            shape_library_lr_mul=shape_library_lr_mul, texture_library_lr_mul=texture_library_lr_mul,
        )

        if planner_type not in ['mlp']:
            raise ValueError(f"Planner type {planner_type} must be in ['mlp']")
        elif planner_type == 'mlp':
            self.planner = PlannerMLP(self.hidden_dim, self.shape_style_dim, self.assembler.action_dim, self.num_actions, lr_mul=planner_lr_mul)
        
        if to_rgb_type not in ['none', 'simple', 'styled']:
            raise ValueError("Invalid to_rgb_type. Expected one of: ['none', 'simple', 'styled']")
        elif to_rgb_type == 'none':
            self.to_rgb = lambda x, noise=None: x[:, :3]
        elif to_rgb_type == 'simple':
            self.to_rgb = nn.Sequential(
                nn.Conv2d(in_channels=self.shape_style_dim, out_channels=3, kernel_size= 1, padding  = 0),
            )
        elif to_rgb_type == 'styled':
            self.output_size = output_size
            self.to_rgb = StyleDecoder(
                size=self.output_size, 
                style_dim=self.shape_style_dim,
                channel_multiplier=2,
                c_out=3,
                c_model=c_model,
                size_in=size_in,
                const_in=const_in,

                seg_classes=(self.num_actions + 1),
            )
        
        
    def sample(self, num_samples, current_device, size, steps=3):
        z = torch.randn(num_samples, self.hidden_dim).to(current_device)
        samples = self.decode(z, size, steps=steps)
        return samples
    
    def render(self, action_list, bg_feat, size, render=True, noise=None):   
        output = {}

        if action_list.shape[1] > 0:
            feature_map, extra_outputs = self.assembler(bg_feat, action_list, size)
            output.update(extra_outputs)
            masks = output['masks']
            textures = output['textures']
            simple_textures = output['simple_textures']
        else:
            bg_texture, bg_simple_texture = self.assembler(bg_feat, action_list, size)
            feature_map = bg_texture
            textures = bg_texture.unsqueeze(-2)
            simple_textures = bg_simple_texture.unsqueeze(-2)
            masks = torch.ones_like(textures[..., 0])
        
        # TODO: Fix this for non "styled" decoders !!!!!!
        feature_map = feature_map.permute(0, 3, 1, 2)
        output['features'] = feature_map
        output['textures'] = textures
        output['simple_textures'] = simple_textures
        output['masks'] = masks
        

        if render:
            with record_function("to_rgb_forward"):
                renders, masks = self.to_rgb(
                    {'textures': simple_textures, 'masks': masks}, 
                    noise=noise,
                )
            
            output['render'] = renders[-1]
            # output['low_res_render'] = renders[0]
            output['low_res_render'] = feature_map[:, 0:3]
            output['segmentation'] = masks

        return output
        
    def decode(self, c, size, steps=3, noise=None, render_every_step=False):
        output = {}
        
        with record_function("planner_forward"):
            bg_feat, action_list = self.planner(c)
        with record_function("postprocess_actions_forward"):
            action_list = self.assembler.postprocess_action(action_list) # Apply postprocess here so all renders have the same actions
        output['bg_and_actions'] = (bg_feat, action_list)
        
        if noise is None:
            noise = [torch.zeros_like(x) for x in self.to_rgb.get_noise_like()]
        
        if not render_every_step:
            render = self.render(action_list, bg_feat, size, noise=noise)

            for key in render.keys():
                output[f'renders/{action_list.shape[1]}/{key}'] = render[key]
        else:
            for i in range(action_list.shape[1] + 1):
                render = self.render(action_list[:, :i], bg_feat, size, noise=noise)

                for key in render.keys():
                    output[f'renders/{i}/{key}'] = render[key]
        return output

    def forward(self, x, size, steps=3, noise=None, mode='sample', render_every_step=False):        
        output = {}
        
        if noise is None:
             noise = [torch.zeros_like(x) for x in self.to_rgb.get_noise_like()]

        if mode == 'render':
            if x.dim() != 4 or x.shape[1] != self.shape_style_dim:
                raise ValueError(f"The input shape isn't consistent with [batch_size, {self.shape_style_dim}, H, W]")
            renders, masks = self.to_rgb(x, noise=noise)
            output['render'] = renders[-1]
            output['low_res_render'] = renders[0]
            output['segmentation'] = masks
        elif mode == 'sample':
            c = torch.randn(x.shape[0], self.hidden_dim).to(x.device)
            output['random_sample'] = c
            output.update(self.decode(c, size, steps=steps, noise=noise, render_every_step=render_every_step))
        elif mode == 'decode':
            if x.dim() != 2 or x.shape[-1] != self.hidden_dim:
                raise Exception(f"The input shape isn't consistent with [batch_size, {self.hidden_dim}]. Input shape: {x.shape}")
            c = x
            output['decoded_sample'] = c
            output.update(self.decode(c, size, steps=steps, noise=noise, render_every_step=render_every_step))
        
        return output

class MyRegularizer(nn.Module):
    def __init__(self, feature_dim, noise_dim, output_dim=3, final_activation=None, hidden_dim=256, loss_mode='l1'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.final_activation = final_activation

        self.loss_mode = loss_mode
        
        self.linear_estimator = nn.Sequential(
            EqualLinear(self.feature_dim + self.noise_dim, self.hidden_dim, lr_mul=1, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.hidden_dim, lr_mul=1, activation='fused_lrelu'),
            EqualLinear(self.hidden_dim, self.output_dim, lr_mul=1, activation='fused_lrelu'),

            # EqualLinear(self.feature_dim + self.noise_dim, self.output_dim, lr_mul=1, activation='fused_lrelu'),
        )
    
    def forward(self, features, noise, render, mode='train'):
        if mode not in ['train', 'reg', 'both']:
            raise ValueError(f"Mode must be in ['train', 'reg', 'both']. Invalid value: {mode}")
        
        
        features = F.interpolate(features, (render.shape[2], render.shape[3]), mode='bilinear', align_corners=False)#, antialias=True)
        noise = torch.cat([F.interpolate(x, (render.shape[2], render.shape[3]), mode='bilinear', align_corners=False) for x in noise], dim=1)
        inputs = torch.cat([features, noise], dim=1).permute(0, 2, 3, 1)
        inputs_shape = inputs.shape
        
        linear_estimate = self.linear_estimator(inputs.flatten(end_dim=-2)).view(*inputs_shape[:-1], -1)
        if self.final_activation is not None:
            linear_estimate = self.final_activation(linear_estimate)
        linear_estimate = linear_estimate.permute(0, 3, 1, 2)
        
        if mode == 'train':
            # Train (performed each step)
            if self.loss_mode == 'l2':
                train_loss = F.mse_loss(linear_estimate, render.detach())
            else:
                train_loss = F.l1_loss(linear_estimate, render.detach())

            return train_loss
        elif mode == 'reg':
            # Regularize (performed every n steps) / This loss must be given the generators inputs when calling backwards()
            if self.loss_mode == 'l2':
                reg_loss = F.mse_loss(linear_estimate.detach(), render) 
            else:
                reg_loss = F.l1_loss(linear_estimate.detach(), render) 
            
            return reg_loss
        elif mode == 'both':
            # Both (performed each step)
            if self.loss_mode == 'l2':
                both_loss = F.mse_loss(linear_estimate, render, reduction='none')
            else:
                both_loss = F.l1_loss(linear_estimate, render, reduction='none')

            return both_loss
        elif mode == 'render':
            return linear_estimate



