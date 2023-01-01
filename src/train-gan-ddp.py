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

from einops import rearrange, repeat

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

from networks.stylegan import LipEqualLinear, StyleGANDiscriminator, normalize_2nd_moment
from networks.custom_modules import MultiShapeStyleLinearApproach, MyRegularizer
from training.clevr import CLEVR_v1
from training.cars_real_traffic import CarsRealTraffic
from training.dataset import ImageFolderDataset
from torch_utils.misc import InfiniteSampler
from metrics import metric_main

################################################################################################################

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def save_checkpoint(generator, generator_ema, my_regularizer, discriminator, discriminator_dual, optimizerG, optimizerD, scalars, save_path):
    temp_path = save_path + f'/temp.zip'
    checkpoint_path = save_path + f'/checkpoint.zip'

    training_state = {
        'generator': generator.state_dict(),
        'generator_ema': generator_ema.state_dict(),
        'my_regularizer': my_regularizer.state_dict(),

        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),

        'discriminator': discriminator.state_dict(),
        'discriminator_dual': discriminator_dual.state_dict(),
    }
    training_state.update(scalars)
    
    torch.save(training_state, temp_path)
    os.replace(temp_path, checkpoint_path) # atomic

# ------------------------------------------------------------------------------------------

def d_r1_loss(real_pred, real_img):
    (grad_real,) = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def calc_pl_lengths(features, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn_like(images) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch.autograd.grad(outputs=outputs, inputs=features, create_graph=True, only_inputs=True)[0] # [B, c, H, W]
    return pl_grads.square().sum((2, 3)).mean(dim=1).sqrt()

def calc_pl_lengths_textures(textures, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn_like(images) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch.autograd.grad(outputs=outputs, inputs=textures, create_graph=True, only_inputs=True)[0] # [batch_size, H, W, num_shapes+1, shape_style_dim] 
    return pl_grads.square().sum((3, 4)).mean((1, 2)).sqrt()


def generate_random_flip_signs(x, prob=0.1):
    return 1 - 2 * (torch.rand_like(x) < prob).float()

def ema_accumulate(model1, model2, beta=0.999):
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.copy_(p2.lerp(p1, beta))
        for b1, b2 in zip(model1.buffers(), model2.buffers()):
            b1.copy_(b2)

def cosine_lerp(a, b, t):
    t = np.clip(t, 0, 1)
    return (a + b) * 0.5 + (a - b) * 0.5 * np.cos(np.pi * t)

def cosine_anneal(
    step, 
    max_val,           # First cycle's max learning rate. 
    min_val,           # Min learning rate

    begin_step,        # Minimum steps before anneal begins
    cycle_steps,       # First cycle step size.
    warmup_steps,      # Linear warmup step size.
    gamma              # Decrease rate of max learning rate by cycle.
):
    if step < begin_step:
        return max_val
    elif step < begin_step + cycle_steps:
        t = (step - begin_step) / cycle_steps
        return cosine_lerp(max_val, min_val, t)
    else:
        step = step - begin_step
        cycle_counter = step // cycle_steps
        cur_max_val = np.maximum(max_val * (gamma ** cycle_counter), min_val)
        step = step % cycle_steps
        if step < warmup_steps:
            t = (step) / warmup_steps
            return min_val + t * (cur_max_val - min_val)
        else:
            t = (step - warmup_steps) / (cycle_steps - warmup_steps)
            return cosine_lerp(cur_max_val, min_val, t)    

def get_metric(metric, generator, c):
    dataset = get_dataset(c)
    dataset_kwargs = EasyDict()
    dataset_kwargs.dataset = dataset
    dataset_kwargs.name = dataset.name
    dataset_kwargs.data_path = c.data_path
    dataset_kwargs.size = c.G_args.output_size

    class ModelWrapper(nn.Module):
        def __init__(self, generator, c):
            super().__init__()
            self.generator = generator
            self.c = c

            self.z_dim = self.generator.hidden_dim
            self.c_dim = 0

        def forward(self, z, c, **kwargs):
            noise_generator = torch.randn if self.c.use_noise else torch.zeros
            input_noise = [noise_generator(z.shape[0], *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
            generator_output = self.generator(z, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='decode')
            output = generator_output[f'renders/{self.c.steps}/render']
            return output

    generator = ModelWrapper(generator, c)
    
    return metric_main.calc_metric(metric, G=generator, num_gpus=c.world_size, rank=c.rank, device=c.device, dataset_kwargs=dataset_kwargs)


# ------------------------------------------------------------------------------------------

def log_wandb(x, c):
    if c.rank == 0:
        wandb.log(x, step=c.global_step, commit=False)

def to_low_to_high(image, low_size, high_size):
    low_res = F.interpolate(image, (low_size, low_size), mode='bilinear')
    high_res = F.interpolate(low_res, (high_size, high_size), mode='bilinear')
    return high_res

class MyLoss(nn.Module):
    def __init__(self, generator, my_regularizer, discriminator, discriminator_dual, c):
        super().__init__()
        self.generator = generator
        self.my_regularizer = my_regularizer
        self.discriminator = discriminator
        self.discriminator_dual = discriminator_dual
        self.c = c
        self.pl_mean = torch.zeros([], device=c.device)
        self.noise_generator = torch.randn if c.use_noise else torch.zeros
        self.scale_mean = torch.ones([], device=c.device) * (np.pi / 4)

        self.planner_grad_norm_mean = torch.zeros([], device=c.device)
        self.assembler_grad_norm_mean = torch.zeros([], device=c.device)
        self.decoder_grad_norm_mean = torch.zeros([], device=c.device)
        
        self.low_size = self.c.feature_volume_size
        self.size = self.c.G_args.output_size

    def accumulate_gradients(self, phase, x, gain, log=True):
        assert phase in ['G', 'G_reg', 'D', 'D_reg']
        batch_size = int(x.shape[0] / 4) * 4
        x = x[:batch_size]

        if phase == "G":
            time_1 = time.time()
            input_noise = [self.noise_generator(batch_size, *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
            generator_output = self.generator(x, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='sample')
            time_2 = time.time()
            samples = generator_output[f'renders/{self.c.steps}/render']
            low_res_samples = generator_output[f'renders/{self.c.steps}/low_res_render']
            masks = generator_output[f'renders/{self.c.steps}/segmentation']
            # masks = generator_output[f'renders/{self.c.steps}/masks'].permute(0, 3, 1, 2)
            # masks = F.interpolate(masks, (self.generator.to_rgb.size, self.generator.to_rgb.size), mode='bilinear', align_corners=False)
            features = generator_output[f'renders/{self.c.steps}/features']


            # Adversarial loss
            # fake_pred = self.discriminator(samples)
            # generator_gan_loss = F.softplus(-fake_pred).mean()
            
            # Segmentation consistency loss
            fake_pred_dual = self.discriminator_dual(torch.cat([samples, F.interpolate(low_res_samples, (self.size, self.size), mode='bilinear')], dim=1))
            generator_dual_loss = F.softplus(-fake_pred_dual).mean()

            # Mask consistency loss
            mask_consistency_loss = torch.zeros([], device=self.c.device)
            original_masks = rearrange(masks[0], 'b H W d -> b d H W')
            for i in range(1, len(masks) - 1):
                low_res_masks = rearrange(masks[i], 'b H W d -> b d H W')
                high_res_masks = rearrange(masks[i + 1], 'b H W d -> b d H W')
                
                if high_res_masks.shape[-1] > original_masks.shape[-1]:
                    high_res_into_low_res_masks = F.interpolate(high_res_masks, (low_res_masks.shape[-2], low_res_masks.shape[-1]), mode='bilinear')
                    mask_consistency_loss += F.mse_loss(low_res_masks, high_res_into_low_res_masks)

            # Segmentation loss 
            ### 2-step renderings
            # bg_feat, action_list = generator_output[f'bg_and_actions']

            # # Randomly remove 1 action from the list
            # random_object = np.random.randint(0, len(action_list))
            # smaller_action_list = action_list[:random_object] + action_list[random_object + 1:]

            # # Render with small action_list
            # small_render = self.generator.render(smaller_action_list, bg_feat, self.c.feature_volume_size, noise=input_noise)
            # small_samples = small_render['render'] # [b, 3, H, W]

            # # On places where mask=0, we want small_samples == samples
            # deleted_mask = masks[:, random_object] # (b, H, W)
            # diffs = F.l1_loss(small_samples, samples, reduction='none')
            # delete_consistency_loss = (
            #     torch.einsum('bdhw,bhw->bd', diffs, 1 - deleted_mask) / repeat(torch.einsum('bhw->b', 1 - deleted_mask) + 1e-6, 'b -> b d', d=3)
            # ).mean()

            # aggregated_colors = torch.einsum('bdhw,bnhw->bnd', samples, masks) / repeat(torch.einsum('bnhw->bn', masks) + 1e-6, 'b n -> b n d', d=3)
            # masked_colors = torch.einsum('bnd,bnhw->bdhw', aggregated_colors, masks)
            # object_consistency_loss = F.mse_loss(samples, masked_colors)

            # shape_alignment_loss = delete_consistency_loss + object_consistency_loss

            ### Apply variance reduction
            # upsampled_features = (torch.cat([
            #     normalize_2nd_moment(F.interpolate(
            #         out, (self.generator.to_rgb.size, self.generator.to_rgb.size), mode='bilinear', align_corners=False,
            #     )) for out in self.generator.to_rgb.outs], dim=1
            # ))
            # d = upsampled_features.shape[1]
            # aggregated_features = torch.einsum('bdhw,bnhw->bnd', upsampled_features, masks) / repeat(torch.einsum('bnhw->bn', masks) + 1e-6, 'b n -> b n d', d=d)
            # masked_features = torch.einsum('bnd,bnhw->bdhw', aggregated_features, masks)
            # shape_alignment_loss = F.mse_loss(upsampled_features, masked_features)

            ### Apply contrastive variance reduction
            # upsampled_features = (torch.cat([
            #     (F.interpolate(
            #         out, (self.generator.to_rgb.size, self.generator.to_rgb.size), mode='bilinear', align_corners=False,
            #     )) for out in self.generator.to_rgb.outs], dim=1
            # ))
            # d, h, w = upsampled_features.shape[1:]
            # total_area = h * w
            # mask_areas = torch.einsum('bnhw->bn', masks)
            # aggregated_features = torch.einsum('bdhw,bnhw->bnd', upsampled_features, masks) / repeat(mask_areas + 1e-6, 'b n -> b n d', d=d)
            # diffs = F.cosine_similarity(
            #     repeat(aggregated_features, 'b n d -> b n d h w', h=h, w=w), 
            #     repeat(upsampled_features, 'b d h w -> b n d h w', n=masks.shape[1]),
            #     dim=2
            # ) # b, n, h, w
            # # Weighted masks are weighted forms of the masks to account for size diffs
            # # ----- mask[i, x, y] = 1 -> weighted_mask[i, x, y] = 1 / area
            # # ----- mask[i, x, y] = 0 -> weighted_mask[i, x, y] = -1 / (total_area - area)
            # weighted_masks = (
            #     masks * repeat(1.0 / (mask_areas + 1e-6) + 1.0 / (total_area - mask_areas + 1e-6), 'b n -> b n h w', h=h, w=w) - 
            #     repeat(1.0 / (total_area - mask_areas + 1e-6), 'b n -> b n h w', h=h, w=w)
            # )
            # shape_alignment_loss = torch.einsum('bnhw,bnhw->b', diffs, weighted_masks).mean()


            # #### Dictionary learning + entropy minimization
            # probs = segmentation
            # d = probs.shape[1]
            # aggregated_probs = torch.einsum('bdhw,bnhw->bnd', probs, (masks + 1e-3)) / repeat(torch.einsum('bnhw->bn', (masks + 1e-3)), 'b n -> b n d', d=d)
            # shape_alignment_loss = compute_entropy(aggregated_probs, dim=1).mean()


            # #### Train the masks to be close to the ARGMAX of the segmentations
            # argmax_segmentations = F.softmax(segmentation.detach() * 1000, dim=1)
            # seg_loss_masks = F.mse_loss(masks, argmax_segmentations).mean()
            # #### Train the segmentations to be close to the masks
            # seg_loss_segs = F.mse_loss(F.softmax(segmentation, dim=1), masks.detach()).mean()

            # seg_pred = self.discriminator_seg(torch.cat([samples.detach(), F.softmax(segmentation, dim=1)], dim=1))
            # masks_pred = self.discriminator_seg(torch.cat([samples.detach(), masks], dim=1))
            # g_seg_loss_seg = F.softplus(seg_pred).mean()
            # g_seg_loss_masks = F.softplus(-masks_pred).mean()
            # generator_seg_loss = g_seg_loss_seg + g_seg_loss_masks

            #### T1 on features
            # decoder_features = torch.cat([
            #     F.interpolate(
            #         out, (self.generator.to_rgb.size, self.generator.to_rgb.size), mode='bilinear', align_corners=False,
            #     ) for out in self.generator.to_rgb.outs], dim=1
            # )

            # linear_estimate = self.my_regularizer(
            #     features, input_noise, 
            #     F.interpolate(samples, (self.c.feature_volume_size, self.c.feature_volume_size), mode='bilinear', align_corners=False),
            #     mode='render'
            # )
            # my_regularizer_loss = self.my_regularizer(
            #     features, input_noise, 
            #     F.interpolate(samples, (self.c.feature_volume_size, self.c.feature_volume_size), mode='bilinear', align_corners=False),
            #     mode='both'
            # )
            # object_masks = torch.sum(masks[:, :-1], dim=1, keepdim=True)
            # my_regularizer_loss = (my_regularizer_loss * object_masks).sum([1, 2]) / (object_masks.sum([1, 2]) + 1e-6)
            # my_regularizer_loss = my_regularizer_loss.mean()

            # generator_seg_loss = shape_alignment_loss
            # generator_loss = generator_gan_loss + generator_seg_loss * self.c.segmentation_weight # + my_regularizer_loss
            # generator_loss = generator_gan_loss * (1 - self.c.segmentation_weight) +  generator_dual_loss * self.c.segmentation_weight + mask_consistency_loss * self.c.segmentation_weight_2
            generator_loss = generator_dual_loss + mask_consistency_loss * self.c.segmentation_weight_2

            time_3 = time.time()
            generator_loss.mean().mul(gain).backward()
            time_4 = time.time()

            if self.c.rank == 0 and log:
                log_wandb({
                    # 'G_loss': generator_gan_loss.item(),
                    # 'G_seg_loss_masks': g_seg_loss_masks.item(),
                    # 'G_seg_loss_seg': g_seg_loss_seg.item(),
                    # 'G_shape_align_loss': shape_alignment_loss.item(),
                    # 'Delete_consistency_loss': delete_consistency_loss.item(),
                    # 'Object_consistency_loss': object_consistency_loss.item(),
                    # 'T1_regularizer_train_G_loss': my_regularizer_loss.item(),
                    'G_dual_loss': generator_dual_loss.item(),
                    'Mask_consistency_loss': mask_consistency_loss.item(),
                    'Timings/G_forward': time_2 - time_1,
                    'Timings/G_loss_computation': time_3 - time_2,
                    'Timings/G_backward': time_4 - time_3,
                }, self.c)
            
            # del generator_output, samples, masks

            # Save running average of gradient norms
            planner_grad_norm = get_grad_norm([self.generator.planner])
            self.planner_grad_norm_mean.copy_(self.planner_grad_norm_mean.lerp(planner_grad_norm.mean(), 0.01).detach())
            
            assembler_grad_norm = get_grad_norm([self.generator.assembler])
            self.assembler_grad_norm_mean.copy_(self.assembler_grad_norm_mean.lerp(assembler_grad_norm.mean(), 0.01).detach())
            
            decoder_grad_norm = get_grad_norm([self.generator.to_rgb])
            self.decoder_grad_norm_mean.copy_(self.decoder_grad_norm_mean.lerp(decoder_grad_norm.mean(), 0.01).detach())
            
            if self.c.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(),   max_norm=self.c.clip_grad_norm * gain)

        if phase == "G_reg":
            input_noise = [self.noise_generator(batch_size, *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
            generator_output = self.generator(x, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='sample')
            regularization_loss = 0

            planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
            # assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
            # decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])            
            
            non_useless_loss = torch.zeros([], device=self.c.device)
            if self.c.non_useless_weight > 0 and self.c.steps > 0:
                masks = generator_output[f'renders/{self.c.steps}/masks'].detach()
                sprites = generator_output[f'renders/{self.c.steps}/sprites']
                non_useless_loss = F.l1_loss(sprites, masks[..., :-1]).mean()
                regularization_loss += (non_useless_loss * self.c.non_useless_weight)
                (non_useless_loss * self.c.non_useless_weight).mean().mul(gain).backward(retain_graph=True)
                # del masks, sprites
                
                if self.c.rank == 0 and log:
                    new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                    # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                    # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])

                    log_wandb( 
                    {
                        'Gradients/non_useless_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                        # 'Gradients/non_useless_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                        # 'Gradients/non_useless_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),
                    }, self.c)

                    planner_grad = new_planner_grad
                    # assembler_grad = new_assembler_grad
                    # decoder_grad = new_decoder_grad
            
            entropy_loss = torch.zeros([], device=self.c.device)
            if self.c.entropy_weight > 0 and self.c.steps > 0:
                # masks = generator_output[f'renders/{self.c.steps}/segmentation'][-1] # Compute entropy on the high-res masks
                masks = generator_output[f'renders/{self.c.steps}/masks']
                entropy_loss = compute_entropy(masks).mean()
                regularization_loss += (entropy_loss * self.c.entropy_weight).mean()
                (entropy_loss * self.c.entropy_weight).mean().mul(gain).backward(retain_graph=True)
                # del masks

                if self.c.rank == 0 and log:
                    new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                    # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                    # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])

                    log_wandb(
                    {
                        'Gradients/entropy_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                        # 'Gradients/entropy_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                        # 'Gradients/entropy_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),
                    }, self.c)

                    planner_grad = new_planner_grad
                    # assembler_grad = new_assembler_grad
                    # decoder_grad = new_decoder_grad

            non_empty_loss = torch.zeros([], device=self.c.device)
            if self.c.non_empty_weight > 0 and self.c.steps > 0:
                sprites = generator_output[f'renders/{self.c.steps}/sprites'].permute(0, 3, 1, 2)   # [B, num_shapes, H, W]
                masks = generator_output[f'renders/{self.c.steps}/masks'].permute(0, 3, 1, 2)       # [B, num_shapes + 1, H, W]
                alphas = generator_output.get(f'renders/{self.c.steps}/actions', {'alpha': None})['alpha'].squeeze() # [B, num_shapes]
                object_masks = masks[:, :-1]
                
                sprite_sizes = sprites.mean([2, 3])     # [B, num_shapes]
                mask_sizes = object_masks.mean([2, 3])  # [B, num_shapes]
                non_empty_loss = (
                    (F.relu(1 - mask_sizes / self.c.min_object_size) * alphas).sum(dim=-1).mean() + 
                    (F.relu(sprite_sizes / self.c.max_object_size - 1) * alphas).sum(dim=-1).mean()
                )
                
                regularization_loss += (non_empty_loss * self.c.non_empty_weight)
                (non_empty_loss * self.c.non_empty_weight).mean().mul(gain).backward(retain_graph=True)
                # del masks

                if self.c.rank == 0 and log:
                    new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                    # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                    # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])

                    log_wandb(
                    {
                        'Gradients/non_empty_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                        # 'Gradients/non_empty_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                        # 'Gradients/non_empty_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),
                    }, self.c)

                    planner_grad = new_planner_grad
                    # assembler_grad = new_assembler_grad
                    # decoder_grad = new_decoder_grad

            scale_invariance_loss = torch.zeros([], device=self.c.device)
            if self.c.scale_invariance_weight > 0 and self.c.steps > 0:
                sprites = generator_output[f'renders/{self.c.steps}/sprites'].permute(0, 3, 1, 2)
                scales = generator_output.get(f'renders/{self.c.steps}/actions', {'scale': None})['scale'].squeeze() # [B, num_shapes]
                alphas = generator_output.get(f'renders/{self.c.steps}/actions', {'alpha': None})['alpha'].squeeze() # [B, num_shapes]


                unscaled_sprite_sizes = sprites.mean([2, 3]) / (scales ** 2)
                unscaled_sprite_sizes_mean = (unscaled_sprite_sizes * alphas).sum() / (alphas.sum() + 1e-6)
                cur_scale_mean = self.scale_mean.lerp(unscaled_sprite_sizes_mean, 0.01)
                self.scale_mean.copy_(cur_scale_mean.detach()).clamp_max_(1.0)
                scale_invariance_loss = ((unscaled_sprite_sizes - self.scale_mean).square() * alphas).sum() / (alphas.sum() + 1e-6)

                regularization_loss += (scale_invariance_loss * self.c.scale_invariance_weight)
                (scale_invariance_loss * self.c.scale_invariance_weight).mean().mul(gain).backward(retain_graph=True)
                # del sprites, scales

                if self.c.rank == 0 and log:
                    new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                    # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                    # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])

                    log_wandb(
                    {
                        'Gradients/scale_invariance_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                        # 'Gradients/scale_invariance_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                        # 'Gradients/scale_invariance_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),

                        'Scale_mean': self.scale_mean.item(),
                    }, self.c)

                    planner_grad = new_planner_grad
                    # assembler_grad = new_assembler_grad
                    # decoder_grad = new_decoder_grad

            t1_loss = torch.zeros([], device=self.c.device)
            if False and self.c.t1_weight > 0 and self.c.steps > 0:
                samples = generator_output[f'renders/{self.c.steps}/render']

                masks = generator_output[f'renders/{self.c.steps}/masks']
                masks = F.interpolate(rearrange(masks, 'b H W s -> b s H W'), (self.generator.to_rgb.size, self.generator.to_rgb.size), mode='bilinear', align_corners=False)
                textures = generator_output[f'renders/{self.c.steps}/textures']

                num_textures = textures.shape[-2]
                textures = F.interpolate(rearrange(textures, 'b H W s d -> b (s d) H W'), (self.generator.to_rgb.size, self.generator.to_rgb.size), mode='bilinear')
                textures = rearrange(textures, 'b (s d) H W -> b d s H W', H=self.generator.to_rgb.size, W=self.generator.to_rgb.size, s=num_textures)
                
                features = (masks[:, None] * textures).sum(dim=2)

                t1_loss = self.my_regularizer(
                    features.detach(), input_noise, 
                    samples, 
                    mode='reg'
                ).mean()

                (t1_loss * self.c.t1_weight).mean().mul(gain).backward(retain_graph=True)
                # del samples, features

                if self.c.rank == 0 and log:
                    new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                    # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                    # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])

                    log_wandb(
                    {
                        'Gradients/t1_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                        # 'Gradients/t1_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                        # 'Gradients/t1_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),
                    }, self.c)

                    planner_grad = new_planner_grad
                    # assembler_grad = new_assembler_grad
                    # decoder_grad = new_decoder_grad
            
            pl_loss = torch.zeros([], device=self.c.device)
            pl_lengths = torch.zeros([], device=self.c.device)
            if self.c.pl_weight > 0.0 and self.c.steps > 0:
                samples = generator_output[f'renders/{self.c.steps}/render']
                textures = generator_output[f'renders/{self.c.steps}/simple_textures']
                
                pl_lengths = calc_pl_lengths_textures(textures, samples)
                cur_pl_mean = self.pl_mean.lerp(pl_lengths.mean(), 0.01)
                self.pl_mean.copy_(cur_pl_mean.detach())
                pl_loss = (pl_lengths - self.pl_mean).square().mean()
                regularization_loss += (pl_loss * self.c.pl_weight)
                (pl_loss * self.c.pl_weight).mean().mul(gain).backward(retain_graph=True)
                # del samples, textures

                if self.c.rank == 0 and log:
                    new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                    # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                    # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])

                    log_wandb(
                    {
                        'Gradients/pl_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                        # 'Gradients/pl_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                        # 'Gradients/pl_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),
                    }, self.c)
                    
                    planner_grad = new_planner_grad
                    # assembler_grad = new_assembler_grad
                    # decoder_grad = new_decoder_grad

            complexity_loss = torch.zeros([], device=self.c.device)
            if (self.c.complexity_max_weight > 0.0 or True) and self.c.steps > 0:
                # Anneal complexity weight
                complexity_weight = cosine_anneal(
                    step=self.c.global_step,
                    max_val=self.c.complexity_max_weight,
                    min_val=self.c.complexity_min_weight,
                    begin_step=self.c.complexity_anneal_start * 1000,
                    cycle_steps=self.c.complexity_anneal_cycle * 1000,
                    warmup_steps=self.c.complexity_anneal_warmup * 1000,
                    gamma=self.c.complexity_anneal_gamma
                )

                extras = generator_output[f'renders/{self.c.steps}/sprite_extras']
                if extras is not None:
                    log_jacobian, warped_coords, non_warped_coords = extras
                    horizontal_smoothness_loss = (
                        (torch.linalg.norm(warped_coords[..., :-1, :, :] - warped_coords[..., 1:, :, :], dim=-1) + 1e-6).log() - 
                        (torch.linalg.norm(non_warped_coords[..., :-1, :, :] - non_warped_coords[..., 1:, :, :], dim=-1) + 1e-6).log()
                    ).abs().flatten(start_dim=1)

                    vertical_smoothness_loss = (
                        (torch.linalg.norm(warped_coords[..., :, :-1, :] - warped_coords[..., :, 1:, :], dim=-1) + 1e-6).log() - 
                        (torch.linalg.norm(non_warped_coords[..., :, :-1, :] - non_warped_coords[..., :, 1:, :], dim=-1) + 1e-6).log()
                    ).abs().flatten(start_dim=1)

                    horizontal_tops = torch.topk(horizontal_smoothness_loss, int(0.01 * horizontal_smoothness_loss.shape[-1]), dim=-1)[0]
                    vertical_tops = torch.topk(vertical_smoothness_loss, int(0.01 * vertical_smoothness_loss.shape[-1]), dim=-1)[0]
                    
                    complexity_loss = horizontal_tops.mean() + vertical_tops.mean()
                    regularization_loss += (complexity_loss * complexity_weight)
                    (complexity_loss * complexity_weight).mean().mul(gain).backward(retain_graph=True)
                    # del extras, warped_coords, non_warped_coords, log_jacobian

                    if self.c.rank == 0 and log:
                        new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                        # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                        # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])
                        
                        log_wandb(
                        {
                            'Gradients/complexity_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                            # 'Gradients/complexity_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                            # 'Gradients/complexity_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),
                        }, self.c)

                        planner_grad = new_planner_grad
                        # assembler_grad = new_assembler_grad
                        # decoder_grad = new_decoder_grad
            
            jacobian_loss = torch.zeros([], device=self.c.device)
            if self.c.jacobian_weight > 0.0 and self.c.steps > 0:
                extras = generator_output[f'renders/{self.c.steps}/sprite_extras']
                if extras is not None:
                    log_jacobian, _, _ = extras
                    jacobian_loss = log_jacobian.abs().mean()
                    regularization_loss += (jacobian_loss * self.c.jacobian_weight)
                    (jacobian_loss * self.c.jacobian_weight).mean().mul(gain).backward(retain_graph=True)
                    # del extras, log_jacobian

                    if self.c.rank == 0 and log:
                        new_planner_grad = torch.cat([param.grad.flatten() for param in self.generator.planner.parameters() if param.grad is not None])
                        # new_assembler_grad = torch.cat([param.grad.flatten() for param in self.generator.assembler.parameters() if param.grad is not None])
                        # new_decoder_grad = torch.cat([param.grad.flatten() for param in self.generator.to_rgb.parameters() if param.grad is not None])
                        
                        log_wandb(
                        {
                            'Gradients/jacobian_reg/planner_grad_norm': torch.linalg.norm(new_planner_grad - planner_grad),
                            # 'Gradients/jacobian_reg/assembler_grad_norm': torch.linalg.norm(new_assembler_grad - assembler_grad),
                            # 'Gradients/jacobian_reg/decoder_grad_norm': torch.linalg.norm(new_decoder_grad - decoder_grad),
                        }, self.c)

                        planner_grad = new_planner_grad
                        # assembler_grad = new_assembler_grad
                        # decoder_grad = new_decoder_grad
            
            # Clip grad norms module by module
            torch.nn.utils.clip_grad_norm_(self.generator.planner.parameters(),   max_norm=self.planner_grad_norm_mean * gain)
            torch.nn.utils.clip_grad_norm_(self.generator.assembler.parameters(), max_norm=self.assembler_grad_norm_mean * gain)
            # torch.nn.utils.clip_grad_norm_(self.generator.to_rgb.parameters(),    max_norm=self.decoder_grad_norm_mean * gain)
            
            
            if self.c.rank == 0 and log:
                wandb.log({
                    'Non_useless_loss': non_useless_loss.item(),
                    'Entropy_loss': entropy_loss.item(),
                    'Non_empty_loss': non_empty_loss.item(),
                    'Scale_invariance_loss': scale_invariance_loss.item(),
                    'T1_loss': t1_loss.item(),
                    'Pl_loss': pl_loss.item(),
                    'Pl_lengths': pl_lengths.mean(),
                    'Complexity_loss': complexity_loss.item(),
                    'Jacobian_loss': jacobian_loss.item(),
                    
                }, step=self.c.global_step, commit=False)
            
            if regularization_loss != 0:
                regularization_loss.mean().mul(0).backward()
            # del generator_output


            if self.c.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(),   max_norm=self.c.clip_grad_norm * gain)

        if phase == "D":
            with torch.no_grad():
                input_noise = [self.noise_generator(batch_size, *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
                generator_output = self.generator(x, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='sample')
                samples = generator_output[f'renders/{self.c.steps}/render']
                low_res_samples = generator_output[f'renders/{self.c.steps}/low_res_render']
                masks = generator_output[f'renders/{self.c.steps}/segmentation']
                # masks = generator_output[f'renders/{self.c.steps}/masks'].permute(0, 3, 1, 2)
                # masks = F.interpolate(masks, (samples.shape[2], samples.shape[3]), mode='bilinear', align_corners=False)
                

            # real_pred = self.discriminator(x)
            # fake_pred = self.discriminator(samples)

            real_pred_dual = self.discriminator_dual(torch.cat([x, to_low_to_high(x, self.low_size, self.size)], dim=1))
            fake_pred_dual = self.discriminator_dual(torch.cat([samples, F.interpolate(low_res_samples, (self.size, self.size), mode='bilinear')], dim=1))

            # d_real_loss = F.softplus(-real_pred).mean()
            # d_fake_loss = F.softplus(fake_pred).mean()
            d_real_dual_loss = F.softplus(-real_pred_dual).mean()
            d_fake_dual_loss = F.softplus(fake_pred_dual).mean()
            # discriminator_loss = d_real_loss + d_fake_loss
            discriminator_loss = d_real_dual_loss + d_fake_dual_loss
            # discriminator_loss = d_real_loss + d_fake_loss + d_real_dual_loss + d_fake_dual_loss
            

            if self.c.rank == 0 and log:
                wandb.log({
                    # "D_loss_real": d_real_loss.item(),
                    # "D_loss_fake": d_fake_loss.item(),
                    
                    "D_dual_loss_real": d_real_dual_loss.item(),
                    "D_dual_loss_fake": d_fake_dual_loss.item(),
                }, step=self.c.global_step, commit=False)
            
            discriminator_loss.mean().mul(gain).backward()
            # del samples

            if self.c.clip_grad_norm > 0:
                # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.c.clip_grad_norm * gain)
                torch.nn.utils.clip_grad_norm_(self.discriminator_dual.parameters(), max_norm=self.c.clip_grad_norm * gain)
                
        if phase == "D_reg":
            with torch.no_grad():
                input_noise = [self.noise_generator(batch_size, *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
                generator_output = self.generator(x, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='sample')
                samples = generator_output[f'renders/{self.c.steps}/render']
                low_res_samples = generator_output[f'renders/{self.c.steps}/low_res_render']
                # segmentation = generator_output[f'renders/{self.c.steps}/segmentation']
            
            # x_tmp = x.detach().requires_grad_(True)
            # real_pred = self.discriminator(x_tmp)
            # x_r1_loss = d_r1_loss(real_pred, x_tmp)
            
            x_dual_tmp = torch.cat([x, to_low_to_high(x, self.low_size, self.size)], dim=1).detach().requires_grad_(True)
            real_pred_dual = self.discriminator_dual(x_dual_tmp)
            dual_r1_loss = d_r1_loss(real_pred_dual, x_dual_tmp)

            # r1_loss = x_r1_loss + dual_r1_loss
            r1_loss =  dual_r1_loss
            d_reg_loss = (r1_loss * self.c.r1_weight).mean()

            if self.c.rank == 0 and log:
                wandb.log({
                    # "R1_loss": x_r1_loss.item(),
                    "R1_dual_loss": dual_r1_loss.item(),
                }, step=self.c.global_step, commit=False)

            d_reg_loss.mean().mul(gain).backward()

            if self.c.clip_grad_norm > 0:
                # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.c.clip_grad_norm * gain)
                torch.nn.utils.clip_grad_norm_(self.discriminator_dual.parameters(), max_norm=self.c.clip_grad_norm * gain)
                

def get_dataset(c):
    if c.data not in ['clevr6', 'real_traffic', 'lsun_beds', 'cartoon_faces', 'tetrominoes', 'multi_dsprites']:
        raise ValueError(f"Dataset {c.data} must be in ['clevr6', 'real_traffic', 'lsun_beds', 'cartoon_faces', 'tetrominoes', 'multi_dsprites']")
    
    dataset = None
    if c.data == 'clevr6':
        dataset = CLEVR_v1('/scratch/ssd004/scratch/amsabour/data/CLEVR_v1.0', train=True, image_size=c.G_args.output_size, num_objects_max=6, num_objects_min=1)
        dataset.name = 'clevr6'
    elif c.data == 'real_traffic':
        dataset = CarsRealTraffic('/scratch/ssd004/scratch/amsabour/data/real_traffic/', image_size=c.G_args.output_size)
        dataset.name = 'real_traffic'
    elif c.data == 'lsun_beds':
        shape_transform = transforms.Compose([
            transforms.Resize(c.G_args.output_size),
            transforms.CenterCrop(c.G_args.output_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = torchvision.datasets.LSUN(
            '/scratch/ssd002/datasets/LSUN/lsun', ['bedroom_train'], 
            transform=shape_transform,
        )

        dataset.name = 'lsun_beds'
    else:
        shape_transform = transforms.Compose([
            transforms.Resize(c.G_args.output_size),
            transforms.CenterCrop(c.G_args.output_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CustomImageFolder(
            c.data_path,
            transform=shape_transform,
        )
        dataset.name = 'Dummy'
    
    # dataset = torchvision.datasets.LSUN(
    #     '/scratch/ssd002/datasets/LSUN/lsun', ['bedroom_train'], 
    #     transform=shape_transform,
    # )

    if len(dataset) > 200_000:
        dataset = torch.utils.data.Subset(dataset, list(range(200_000)))
    
    return dataset

def get_dataloader(c):
    dataset = get_dataset(c)

    sampler = DistributedSampler(dataset, drop_last=True, num_replicas=c.world_size, rank=c.rank)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=c.batch_size // c.world_size, sampler=sampler, num_workers=c.dataloader_workers, pin_memory=True, drop_last=True
    )
    
    return dataloader

def accumulate_gradients(modules, c, grad_norm=1.0):
    world_size = c.world_size
    if not (world_size > 1):
        return 
     
    params = []
    for module in modules:
        params += [param for param in module.parameters() if param.grad is not None]
    
    if len(params) > 0:
        flat = torch.cat([param.grad.flatten() for param in params])
        dist.all_reduce(flat, dist.ReduceOp.SUM)
        flat /= world_size

        torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)        
        
        grads = flat.split([param.numel() for param in params])
        for param, grad in zip(params, grads):
            param.grad = grad.reshape(param.shape)

        del flat, grads
    del params

def plot_training_progress(generator, noise, c):
    noise_generator = torch.randn if c.use_noise else torch.zeros
    size = c.feature_volume_size
    # Plot everything (generated samples, masks, sprites, textures, ....)
    with torch.no_grad():
        input_noise = [noise_generator(8, *_noise.shape[1:], device=c.device) for _noise in generator.to_rgb.get_noise_like()]
        output = generator(torch.randn_like(noise[:8]), size, steps=c.steps, noise=input_noise, mode='decode', render_every_step=True)
        masks = output[f'renders/{c.steps}/masks']
        segmentation = rearrange(output[f'renders/{c.steps}/segmentation'][-1], 'b H W d -> b d H W')
        sprites = output.get(f'renders/{c.steps}/sprites', None)
        centers = output.get(f'renders/{c.steps}/actions', {'translation': None})['translation']

        textures = output.get(f'renders/{c.steps}/textures', None)
        if textures is not None:
            # # Reduce dimensionality of textures using PCA
            # assert textures.dim() == 5
            # b, H, W, n, d = textures.shape
            # textures = rearrange(textures, 'b H W n d -> (b H W n) d')
            
            # from sklearn.decomposition import PCA
            # textures = PCA(n_components=3).fit_transform(textures.detach().cpu().numpy())
            # textures = torch.from_numpy(textures)
            
            # # Rescale everything to [0,1]
            # textures_min = textures.min(dim=-1, keepdim=True)[0]
            # textures_max = textures.max(dim=-1, keepdim=True)[0]
            # textures = (textures - textures_min) / (textures_max - textures_min + 1e-6)

            # textures = rearrange(textures, '(b H W n) d -> b H W n d', b=b, H=H, W=W, n=n)

            textures = torch.clamp(textures * 0.5 + 0.5, 0, 1)

        num_rows, num_cols = 1, 2
        w, h = num_rows * 6, num_cols * (c.steps + 1)
        fig, axes = plt.subplots(w, h, figsize=(h * 3, w * 3), constrained_layout=True)

        for idx in range(num_rows * num_cols):
            row_idx = idx // num_cols
            col_idx = idx % num_cols

            for k in range(c.steps + 1):
                # Renders
                axes[row_idx * 5 + 0, col_idx * (c.steps + 1) + k].imshow(torch.clamp((output[f'renders/{k}/render'].permute(0, 2, 3, 1) + 1) / 2, 0, 1)[idx].cpu().detach().numpy())
                axes[row_idx * 5 + 0, col_idx * (c.steps + 1) + k].title.set_text(f'Render @ Step {k}')

                # Low-res renders
                axes[row_idx * 5 + 1, col_idx * (c.steps + 1) + k].imshow(torch.clamp((output[f'renders/{k}/low_res_render'].permute(0, 2, 3, 1) + 1) / 2, 0, 1)[idx].cpu().detach().numpy())
                axes[row_idx * 5 + 1, col_idx * (c.steps + 1) + k].title.set_text(f'Low-res Render @ Step {k}')

                # Sprites
                if k > 0:
                    axes[row_idx * 5 + 2, col_idx * (c.steps + 1) + k].imshow((sprites)[idx, ..., k - 1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 5 + 2, col_idx * (c.steps + 1) + k].title.set_text(f'Sprite @ Step {k}')
                    
                    center_x, center_y = centers[idx, k - 1, 0:2].detach().cpu().numpy()
                    center = ((center_y + 1) / 2 * size, (center_x + 1) / 2 * size)
                    axes[row_idx * 5 + 2, col_idx * (c.steps + 1) + k].add_patch(Circle(center, 1, color='red'))

                # Masks
                if k > 0:
                    axes[row_idx * 5 + 3, col_idx * (c.steps + 1) + k].imshow((masks)[idx, ..., k-1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 5 + 3, col_idx * (c.steps + 1) + k].title.set_text(f'Mask @ Step {k}')
                else:
                    axes[row_idx * 5 + 3, col_idx * (c.steps + 1) + k].imshow((masks)[idx, ..., -1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 5 + 3, col_idx * (c.steps + 1) + k].title.set_text(f'Mask @ Step {k}')

                # Segmentation
                if k > 0:
                    axes[row_idx * 5 + 4, col_idx * (c.steps + 1) + k].imshow((segmentation)[idx, k-1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 5 + 4, col_idx * (c.steps + 1) + k].title.set_text(f'Mask @ Step {k}')
                else:
                    axes[row_idx * 5 + 4, col_idx * (c.steps + 1) + k].imshow((segmentation)[idx, -1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 5 + 4, col_idx * (c.steps + 1) + k].title.set_text(f'Mask @ Step {k}')

                # Texture
                if k > 0:
                    axes[row_idx * 5 + 5, col_idx * (c.steps + 1) + k].imshow((textures[idx, ..., k-1, :3]).cpu().detach().numpy())
                    axes[row_idx * 5 + 5, col_idx * (c.steps + 1) + k].title.set_text(f'Texture @ Step {k}')
                else:
                    axes[row_idx * 5 + 5, col_idx * (c.steps + 1) + k].imshow((textures[idx, ..., -1, :3]).cpu().detach().numpy())
                    axes[row_idx * 5 + 5, col_idx * (c.steps + 1) + k].title.set_text(f'Texture @ Step {k}')

        wandb.log({"Construction timeline": fig}, step=c.global_step)
        plt.clf()
        plt.close()
        del output, masks, sprites, textures, centers

    # Plot samples (generated samples, masks, sprites, textures, ....)
    with torch.no_grad():
        samples = []
        for j in range(8):
            input_noise = 'const' if c.use_noise else None
            generator_output = generator(noise[j * 8: j * 8 + 8], size, steps=c.steps, noise=input_noise, mode='decode')
            samples.append(
                torch.clamp((generator_output[f'renders/{c.steps}/render'] + 1) / 2, 0, 1).permute(0, 2, 3, 1).cpu()
            )
        samples = torch.cat(samples, dim=0).numpy()
        plot_samples(samples, 8, 8, show=False)
        wandb.log({'Generated Samples': plt}, step=c.global_step)
        plt.clf()
        plt.close()
        del samples

    gc.collect()
    for _ in range(5):
        torch.cuda.empty_cache()       

    import os, psutil; print(f"Step {c.global_step}:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, flush=True)

def training_loop(c):
    torch.backends.cudnn.benchmark = True               # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.


    print(torch.cuda.nccl.version(), flush=True)

    # Load training data
    dataloader= get_dataloader(c)

    # Instantiate generator and discriminator
    generator = MultiShapeStyleLinearApproach(**c.G_args).to(c.device)
    generator_ema = MultiShapeStyleLinearApproach(**c.G_args).to(c.device).eval()
    generator_ema.load_state_dict(generator.state_dict())

    my_regularizer = MyRegularizer(c.G_args.shape_style_dim, generator.to_rgb.num_layers, output_dim=3, final_activation=None, loss_mode='l2').to(c.device)
    discriminator = StyleGANDiscriminator(c.G_args.output_size, in_channels=3, channel_multiplier=1).to(c.device) 
    discriminator_dual = StyleGANDiscriminator(c.G_args.output_size, in_channels=6, channel_multiplier=1).to(c.device)

    # Setup loss and optimizers
    loss = MyLoss(generator, my_regularizer, discriminator, discriminator_dual, c)

    g_mb_ratio = (c.g_reg_every) / (c.g_reg_every + 1)
    optimizerG = torch.optim.Adam(
        [
            {'params': generator.parameters()},
            {'params': my_regularizer.parameters()},
        ],
        lr=c.generator_lr * g_mb_ratio, betas=(0.0, 0.99 ** g_mb_ratio), eps=1e-8
    )

    d_mb_ratio = (c.d_reg_every) / (c.d_reg_every + 1)
    optimizerD = torch.optim.Adam(
        [
            {'params': discriminator.parameters()},
            {'params': discriminator_dual.parameters()},
        ],
        lr=c.discriminator_lr * d_mb_ratio, betas=(0.0, 0.99 ** d_mb_ratio), eps=1e-8
    )

    # Distribute across GPUs.
    if c.rank == 0:
        print(f'Distributing across {c.world_size} GPUs...')
    
    # Resume from checkpoint
    if os.path.exists(c.resume):
        if c.rank == 0:
            print(f'Checkpoint found. Resuming...')
        
        loaded_dict = torch.load(c.resume)
        c.global_step = int(loaded_dict['global_step'])
        loss.pl_mean = torch.tensor(float(loaded_dict['pl_mean']), device=loss.c.device) 
        loss.scale_mean = torch.tensor(float(loaded_dict['scale_mean']), device=loss.c.device)

        generator.load_state_dict(loaded_dict['generator'])
        generator_ema.load_state_dict(loaded_dict['generator_ema'])
        my_regularizer.load_state_dict(loaded_dict['my_regularizer'])
        discriminator.load_state_dict(loaded_dict['discriminator'])

        if 'discriminator_dual' in loaded_dict.keys():
            discriminator_dual.load_state_dict(loaded_dict['discriminator_dual'])
        if 'optimizerG' in loaded_dict.keys():
            optimizerG.load_state_dict(loaded_dict['optimizerG'])
        if 'optimizerD' in loaded_dict.keys():
            optimizerD.load_state_dict(loaded_dict['optimizerD'])
            
    else:
        print(f'No checkpoint found. Starting from scratch...')
    
    if c.world_size > 1:
        for module in [generator, generator_ema, my_regularizer, discriminator, discriminator_dual]:
            if module is not None:
                for param in params_and_buffers(module):
                    dist.broadcast(param, src=0)
    
    # Setup logging
    if c.rank == 0:
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project="my-test-project", entity="amsabour", resume="allow", dir=c.save_dir , id=c.job_id)
        wandb.config.update(c, allow_val_change=True)
        wandb.watch(generator, log_freq=500)
    

    noise = torch.randn(64, c.G_args.hidden_dim).to(c.device)
    
    while c.global_step < c.kimg * 1000:
        for i, x in enumerate(dataloader):
            if not isinstance(x, torch.Tensor):
                x = x[0]
                assert isinstance(x, torch.Tensor)

            c.global_step += c.batch_size

            start_fetch_data = time.time()
            x = x.to(c.device, non_blocking=True)
            # x = (x.to(c.device).to(torch.float32) / 127.5 - 1) # Scale to [-1, 1]
            # x = F.interpolate(x, (c.G_args.output_size, c.G_args.output_size), mode='bilinear', align_corners=False)
            end_fetch_data = time.time()
            log_wandb({'Timings/data_fetch': end_fetch_data - start_fetch_data}, c)

            # G Step
            if i % 1 == 0:
                generator.requires_grad_(True)
                my_regularizer.requires_grad_(True)
                discriminator.requires_grad_(False)
                discriminator_dual.requires_grad_(False)

                optimizerG.zero_grad(set_to_none=True)
                start_G = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0): 
                    loss.accumulate_gradients('G', xx, gain=1 / c.num_batch_splits, log=(i % 1 == 0))
                end_G = time.time()
                accumulate_gradients([generator], c, grad_norm=c.clip_grad_norm)
                log_wandb(
                    {
                        'Gradients/G_grad_norm': get_grad_norm([generator]),
                        
                        'Gradients/planner_grad_norm': get_grad_norm([generator.planner]),

                        'Gradients/assembler_grad_norm': get_grad_norm([generator.assembler]),
                        # 'Gradients/to_action_grad_norm': get_grad_norm([generator.assembler.to_action]),
                        'Gradients/shape_library_grad_norm': get_grad_norm([generator.assembler.shape_library]),
                        'Gradients/texture_library_grad_norm': get_grad_norm([generator.assembler.texture_library]),
                        
                        'Gradients/decoder_grad_norm': get_grad_norm([generator.to_rgb]),

                        'Gradients/regularizer_grad_norm': get_grad_norm([my_regularizer]),
                    }, c)

                end_G_acc = time.time()
                optimizerG.step()
                end_G_opt = time.time()
                log_wandb({
                    'Timings/G_forward_backward': end_G - start_G,
                    'Timings/G_accumulate': end_G_acc - end_G,
                    'Timings/G_opt': end_G_opt - end_G_acc
                }, c)
                
                
            # G reg step
            if i % c.g_reg_every == 0:
                generator.requires_grad_(True)
                my_regularizer.requires_grad_(False)
                discriminator.requires_grad_(False)
                discriminator_dual.requires_grad_(False)
                
                optimizerG.zero_grad(set_to_none=False)
                start_G_reg = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0): 
                    loss.accumulate_gradients('G_reg', xx, gain=c.g_reg_every / c.num_batch_splits, log=True)
                end_G_reg = time.time()
                accumulate_gradients([generator], c, grad_norm=c.clip_grad_norm * c.g_reg_every)
                log_wandb(
                    {
                        'Gradients/G_reg_grad_norm': get_grad_norm([generator]),
                        
                        'Gradients/planner_reg_grad_norm': get_grad_norm([generator.planner]),

                        'Gradients/assembler_reg_grad_norm': get_grad_norm([generator.assembler]),
                        # 'Gradients/to_action_reg_grad_norm': get_grad_norm([generator.assembler.to_action]),
                        'Gradients/shape_library_reg_grad_norm': get_grad_norm([generator.assembler.shape_library]),
                        'Gradients/texture_library_reg_grad_norm': get_grad_norm([generator.assembler.texture_library]),
                        
                        'Gradients/decoder_reg_grad_norm': get_grad_norm([generator.to_rgb]),
                    }, c)

                end_G_reg_acc = time.time()
                optimizerG.step()
                end_G_reg_opt = time.time()

                log_wandb({
                    'Timings/G_reg_forward_backward': end_G_reg - start_G_reg,
                    'Timings/G_reg_accumulate': end_G_reg_acc - end_G_reg,
                    'Timings/G_reg_opt': end_G_reg_opt - end_G_reg_acc,
                }, c)

            # D Step
            if i % 1 == 0:
                generator.requires_grad_(False)
                my_regularizer.requires_grad_(False)
                discriminator.requires_grad_(True)
                discriminator_dual.requires_grad_(True)

                optimizerD.zero_grad(set_to_none=True)
                start_D = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0): 
                    loss.accumulate_gradients('D', xx, gain=1 / c.num_batch_splits, log=(i % 1 == 0))
                end_D = time.time()
                accumulate_gradients([discriminator, discriminator_dual], c, grad_norm=c.clip_grad_norm)
                log_wandb(
                    {
                        'Gradients/D_grad_norm': get_grad_norm([discriminator]),
                        'Gradients/D_dual_grad_norm': get_grad_norm([discriminator_dual]),
                    }, c)

                end_D_acc = time.time()
                optimizerD.step()
                end_D_opt = time.time()

                log_wandb({
                    'Timings/D_forward_backward': end_D - start_D,
                    'Timings/D_accumulate': end_D_acc - end_D,
                    'Timings/D_opt': end_D_opt - end_D_acc
                }, c)
                
            # D reg step
            if i % c.d_reg_every == 0:
                generator.requires_grad_(False)
                my_regularizer.requires_grad_(False)
                discriminator.requires_grad_(True)
                discriminator_dual.requires_grad_(True)

                optimizerD.zero_grad()
                start_D_reg = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0):
                    loss.accumulate_gradients('D_reg', xx, gain=c.d_reg_every / c.num_batch_splits, log=True)
                end_D_reg = time.time()
                accumulate_gradients([discriminator, discriminator_dual], c, grad_norm=c.clip_grad_norm * c.d_reg_every)
                log_wandb(
                    {
                        'Gradients/D_reg_grad_norm': get_grad_norm([discriminator]),
                        'Gradients/D_dual_reg_grad_norm': get_grad_norm([discriminator_dual]),
                    }, c)
                end_D_reg_acc = time.time()    
                optimizerD.step()
                end_D_reg_opt = time.time()

                log_wandb({
                    'Timings/D_reg_forward_backward': end_D_reg - start_D_reg,
                    'Timings/D_reg_accumulate': end_D_reg_acc - end_D_reg,
                    'Timings/D_reg_opt': end_D_reg_opt - end_D_reg_acc
                }, c)

            # Update progressive librarires
            start_progressive = time.time()
            generator.assembler.shape_library.step(c.batch_size / (c.shape_progressive_unlock_kimg * 1000))
            generator.assembler.sprite_and_texture_library.step(c.batch_size / (c.shape_progressive_unlock_kimg * 1000))
            generator.assembler.texture_library.step(c.batch_size / (c.texture_progressive_unlock_kimg * 1000))
            
            end_progressive = time.time()
            log_wandb({'Timings/progressive_update': end_progressive - start_progressive} ,c)

            # Update model_ema
            start_ema = time.time()
            ema_nimg = (c.batch_size / 32) * (c.ema_kimg * 1000)
            if c.ema_rampup is not None:
                ema_nimg = min(ema_nimg, c.global_step * c.ema_rampup)
            ema_beta = 0.5 ** (c.batch_size / max(ema_nimg, 1e-8))
            ema_accumulate(generator_ema, generator, ema_beta)
            end_ema = time.time()
            log_wandb({'Timings/ema_update': end_ema - start_ema}, c)

            if c.rank == 0:
                wandb.log({"commited": True}, step=c.global_step)
            
            # Plotting / Saving / Computing metrics
            if c.rank == 0:
                if (c.global_step % (c.tick * 1000)) < c.batch_size:
                    print(f"Step {c.global_step}", flush=True)

                # Plot samples and reconstructions
                if (c.global_step % (c.plot * 1000)) < c.batch_size  or c.global_step == c.batch_size:
                    plot_training_progress(generator_ema, noise, c)
                    
                # Save checkpoint
                if (c.global_step % (c.snap * 1000)) < c.batch_size:
                    save_checkpoint(
                        generator, generator_ema, my_regularizer, discriminator, discriminator_dual, optimizerG, optimizerD, 
                        {'global_step': c.global_step, 'pl_mean': loss.pl_mean.item(), 'scale_mean': loss.scale_mean.item()}, 
                        c.save_dir
                    )

            if (c.global_step % (c.measure * 1000)) < c.batch_size:
                # Broadcast generator_ema
                for param in params_and_buffers(generator_ema):
                    dist.broadcast(param, src=0)
                
                metric_result = get_metric('fid50k_full', generator_ema, c) # Compute metric in parallel

                if c.rank == 0:
                    wandb.log(metric_result['results'], step=c.global_step + 1)

################################################################################################################


@click.command()
# Required.
@click.option('--save-dir',             help='Where to save the results', metavar='DIR',                           required=True)
@click.option('--data',                 help='Path to training data folder',                                       type=str, required=True)
@click.option('--batch',                help='Total batch size', metavar='INT',                                    type=click.IntRange(min=1), required=True)
@click.option('--output-size',          help='Resolution of samples generated by the generator',                   type=click.Choice(['16', '32', '64', '128', '256']), required=True)
@click.option('--feature-volume-size',  help='Resolution of the feature volume', metavar='INT',                    type=click.IntRange(min=16), required=True)
@click.option('--num-batch-splits',     help='Number of batches to accumulate per step', metavar='INT',            type=click.IntRange(min=1), required=True)
@click.option('--d-reg-every',          help='Dicsriminator regularization every n batches', metavar='INT',        type=click.IntRange(min=1), required=True)
@click.option('--r1-weight',            help='R1 regularization weight', metavar='FLOAT',                          type=click.FloatRange(min=0), required=True)
@click.option('--g-reg-every',          help='Generator regularization every n batches', metavar='INT',            type=click.IntRange(min=1), required=True)
@click.option('--t1-weight',            help='T1 regularization weight', metavar='FLOAT',                          type=click.FloatRange(min=0), required=True)
  
# Optional features.  
@click.option('--resume',               help='Resume from given network pickle',  metavar='[PATH|URL]',  type=str)
@click.option('--entropy-weight',       help='Entropy regularization weight',     metavar='FLOAT',        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--non-useless-weight',   help='Non useless regularization weight', metavar='FLOAT',        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--non-empty-weight',     help='Non empty regularization weight',   metavar='FLOAT',        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--scale-invariance-weight', help='Scale invariance regularization weight',   metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--min-object-size',      help='Minimum object size',               metavar='FLOAT',        type=click.FloatRange(min=0, max=1), default=0.05, show_default=True)
@click.option('--max-object-size',      help='Maximum object size',               metavar='FLOAT',        type=click.FloatRange(min=0, max=1), default=0.75, show_default=True)
@click.option('--pl-weight',            help='Path length regularization weight', metavar='FLOAT',        type=click.FloatRange(min=0), default=2, show_default=True)
@click.option('--jacobian-weight',      help='Sprite unit jacobian weight',       metavar='FLOAT',        type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--segmentation-weight',  help='Segmentation regularization weight', metavar='FLOAT',       type=click.FloatRange(min=0, max=1), required=True)
@click.option('--segmentation-weight-2',  help='Segmentation regularization weight', metavar='FLOAT',       type=click.FloatRange(min=0), required=True)


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
@click.option('--measure',      help='How often to measure metrics', metavar='KIMG',            type=click.IntRange(min=1), default=50, show_default=True)
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
    c.data = str(opts.data).lower()
    c.data_path = opts.data
    c.batch_size = opts.batch
    if c.batch_size % c.world_size != 0:
        raise ValueError(f"[SANITY CHECK FAILED] The batch size {c.batch_size} is not divisible by the number of replicas {c.world_size}")
    c.num_batch_splits = opts.num_batch_splits

    c.d_reg_every = opts.d_reg_every
    c.r1_weight = opts.r1_weight
    
    c.g_reg_every = opts.g_reg_every
    c.t1_weight = opts.t1_weight
    c.segmentation_weight = opts.segmentation_weight
    c.segmentation_weight_2 = opts.segmentation_weight_2
    c.entropy_weight = opts.entropy_weight
    c.non_useless_weight = opts.non_useless_weight
    c.non_empty_weight = opts.non_empty_weight
    c.scale_invariance_weight = opts.scale_invariance_weight
    c.min_object_size = opts.min_object_size
    c.max_object_size = opts.max_object_size
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
    
    c.kimg = opts.kimg
    c.tick = opts.tick
    c.plot = opts.plot
    c.snap = opts.snap
    c.measure = opts.measure
    c.dataloader_workers = opts.workers

    c.global_step = 0
    c.resume = c.save_dir + f'/checkpoint.zip'
    
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