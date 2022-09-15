################################ IMPORTS ################################################################################
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


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from networks.stylegan import LipEqualLinear, StyleGANDiscriminator
from networks.custom_modules import MultiShapeStyleLinearApproach, MyRegularizer

################################################################################################################

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def save_checkpoint(generator, generator_ema, my_regularizer, discriminator, global_step, save_path):
    torch.save(
        {
            'global_step': global_step,
            'generator': generator.state_dict(),
            'generator_ema': generator_ema.state_dict(),
            'my_regularizer': my_regularizer.state_dict(),
            'discriminator': discriminator.state_dict(),
        }, save_path
    )

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
    return pl_grads.square().sum((1, 2, 3)).mean(dim=1).sqrt()


def generate_random_flip_signs(x, prob=0.1):
    return 1 - 2 * (torch.rand_like(x) < prob).float()

def ema_accumulate(model1, model2, beta=0.999):
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.copy_(p2.lerp(p1, beta))
        for b1, b2 in zip(model1.buffers(), model2.buffers()):
            b1.copy_(b2)

# ------------------------------------------------------------------------------------------

def log_wandb(x, c):
    if c.rank == 0:
        wandb.log(x, step=c.global_step, commit=False)

class MyLoss(nn.Module):
    def __init__(self, generator, my_regularizer, discriminator, c):
        super().__init__()
        self.generator = generator
        self.my_regularizer = my_regularizer
        self.discriminator = discriminator
        self.c = c
        self.pl_mean = torch.zeros([], device=c.device)
        self.noise_generator = torch.randn if c.use_noise else torch.zeros

    def accumulate_gradients(self, phase, x, gain, log=True):
        assert phase in ['G', 'G_reg', 'D', 'D_reg']
        batch_size = x.shape[0]

        if phase == "G":
            input_noise = [self.noise_generator(batch_size, *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
            generator_output = self.generator(x, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='sample')
            samples = generator_output[f'renders/{self.c.steps}/render']
            features = generator_output[f'renders/{self.c.steps}/features']

            fake_pred = self.discriminator(samples)
            generator_gan_loss = F.softplus(-fake_pred).mean()
            my_regularizer_loss = self.my_regularizer(
                F.interpolate(features, (self.c.G_args.output_size, self.c.G_args.output_size), mode='bilinear', align_corners=False).detach(), input_noise, 
                samples, 
                mode='train'
            ).mean()
            generator_loss = generator_gan_loss + my_regularizer_loss

            generator_loss.mean().mul(gain).backward()

            if self.c.rank == 0 and log:
                wandb.log({
                    'G_loss': generator_gan_loss.item(),
                    'T1_regularizer_train_G_loss': my_regularizer_loss.item(),
                }, step=self.c.global_step, commit=False)
            
            del generator_output, samples, features

        if phase == "G_reg":
            input_noise = [self.noise_generator(batch_size, *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
            generator_output = self.generator(x, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='sample')
            regularization_loss = 0
            
            non_useless_loss = torch.zeros([], device=self.c.device)
            if self.c.non_useless_weight > 0:
                masks = generator_output[f'renders/{self.c.steps}/masks'].detach()
                sprites = generator_output[f'renders/{self.c.steps}/sprites']
                non_useless_loss = F.l1_loss(sprites, masks[..., :-1]).mean()
                regularization_loss += (non_useless_loss * self.c.non_useless_weight)
                del masks, sprites

            entropy_loss = torch.zeros([], device=self.c.device)
            if self.c.entropy_weight > 0:
                masks = generator_output[f'renders/{self.c.steps}/masks']
                entropy_loss = compute_entropy(masks).mean()
                regularization_loss += (entropy_loss * self.c.entropy_weight).mean()
                del masks

            non_empty_loss = torch.zeros([], device=self.c.device)
            if self.c.non_empty_weight > 0:
                sprites = generator_output[f'renders/{self.c.steps}/sprites']
                sprites_areas = sprites.sum((1, 2))
                scales = generator_output[f'renders/{self.c.steps}/actions']['scale'].squeeze() # [batch_size, steps]
                scaled_sprites_areas = sprites_areas / (scales ** 2)
                non_empty_loss = (scaled_sprites_areas - 1).square().mean()
                regularization_loss += (non_empty_loss * self.c.non_empty_weight)
                del sprites, scales

            t1_loss = torch.zeros([], device=self.c.device)
            if self.c.t1_weight > 0:
                samples = generator_output[f'renders/{self.c.steps}/render']
                features = generator_output[f'renders/{self.c.steps}/features']
                t1_loss = self.my_regularizer(
                    F.interpolate(features, (self.c.G_args.output_size, self.c.G_args.output_size), mode='bilinear', align_corners=False).detach(), input_noise, 
                    samples, 
                    mode='reg'
                ).mean()
                regularization_loss += (t1_loss * self.c.t1_weight).mean()
                del samples, features

            pl_loss = torch.zeros([], device=self.c.device)
            pl_lengths = torch.zeros([], device=self.c.device)
            if self.c.pl_weight > 0.0:
                samples = generator_output[f'renders/{self.c.steps}/render']
                textures = generator_output[f'renders/{self.c.steps}/textures']
                
                pl_lengths = calc_pl_lengths_textures(textures, samples)
                cur_pl_mean = self.pl_mean.lerp(pl_lengths.mean(), 0.01)
                self.pl_mean.copy_(cur_pl_mean.detach())
                pl_loss = (pl_lengths - self.pl_mean).square().mean()
                regularization_loss += (pl_loss * self.c.pl_weight)
                del samples, textures
            
            complexity_loss = torch.zeros([], device=self.c.device)
            if self.c.complexity_weight > 0.0:
                extras = generator_output[f'renders/{self.c.steps}/extras']
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
                
                complexity_loss = horizontal_tops.mean() + vertical_tops.mean() + log_jacobian.abs().mean()
                regularization_loss += (complexity_loss * self.c.complexity_weight)
            
            if self.c.rank == 0 and log:
                wandb.log({
                    'Non_useless_loss': non_useless_loss.item(),
                    'Entropy_loss': entropy_loss.item(),
                    'Non_empty_loss': non_empty_loss.item(),
                    'T1_loss': t1_loss.item(),
                    'Pl_loss': pl_loss.item(),
                    'Pl_lengths': pl_lengths.mean(),
                    'Complexity_loss': complexity_loss.item(),
                }, step=self.c.global_step, commit=False)
            
            if regularization_loss != 0:
                regularization_loss.mean().mul(gain).backward()
            del generator_output

        if phase == "D":
            with torch.no_grad():
                input_noise = [self.noise_generator(batch_size, *_noise.shape[1:], device=self.c.device) for _noise in self.generator.to_rgb.get_noise_like()]
                samples = self.generator(x, self.c.feature_volume_size, steps=self.c.steps, noise=input_noise, mode='sample')[f'renders/{self.c.steps}/render']

            real_pred = self.discriminator(x)
            fake_pred = self.discriminator(samples)

            random_flip_prob = self.c.random_flip_prob * max(0.0, 1 - (self.c.global_step) / (self.c.random_flip_kimg * 1000))
            d_real_loss = F.softplus(-real_pred * generate_random_flip_signs(real_pred, random_flip_prob)).mean()
            d_fake_loss = F.softplus(fake_pred * generate_random_flip_signs(fake_pred, random_flip_prob)).mean()
            discriminator_loss = d_real_loss + d_fake_loss

            if self.c.rank == 0 and log:
                wandb.log({
                    "D_loss_real": d_real_loss.item(),
                    "D_loss_fake": d_fake_loss.item(),
                }, step=self.c.global_step, commit=False)
            
            discriminator_loss.mean().mul(gain).backward()
            del samples

        if phase == "D_reg":
            x_tmp = x.detach().requires_grad_(True)
            real_pred = self.discriminator(x_tmp)
            r1_loss = d_r1_loss(real_pred, x_tmp)
            d_reg_loss = (r1_loss * self.c.r1_weight).mean()

            if self.c.rank == 0 and log:
                wandb.log({
                    "R1_loss": r1_loss.item()
                }, step=self.c.global_step, commit=False)

            d_reg_loss.mean().mul(gain).backward()

def get_dataloader(c):
    # Setup dataloader
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

    sampler = DistributedSampler(dataset, drop_last=True, num_replicas=c.world_size, rank=c.rank)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=c.batch_size // c.world_size, sampler=sampler, num_workers=c.dataloader_workers, pin_memory=True,
    )

    return dataloader

def accumulate_gradients(modules, c):
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

    for module in modules:
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)

def plot_training_progress(generator, noise, c):
    noise_generator = torch.randn if c.use_noise else torch.zeros
    size = c.feature_volume_size
    # Plot everything (generated samples, masks, sprites, textures, ....)
    with torch.no_grad():
        input_noise = [noise_generator(8, *_noise.shape[1:], device=c.device) for _noise in generator.to_rgb.get_noise_like()]
        output = generator(torch.randn_like(noise[:8]), size, steps=c.steps, noise=input_noise, mode='decode', render_every_step=True)
        masks = output[f'renders/{c.steps}/masks']
        sprites = output[f'renders/{c.steps}/sprites']
        textures = output[f'renders/{c.steps}/textures']
        centers = output[f'renders/{c.steps}/actions']['translation']

        num_rows, num_cols = 1, 2
        w, h = num_rows * 4, num_cols * (c.steps + 1)
        fig, axes = plt.subplots(w, h, figsize=(h * 3, w * 3), constrained_layout=True)

        for idx in range(num_rows * num_cols):
            row_idx = idx // num_cols
            col_idx = idx % num_cols

            for k in range(c.steps + 1):
                # Renders
                axes[row_idx * 4 + 0, col_idx * (c.steps + 1) + k].imshow(torch.clamp((output[f'renders/{k}/render'].permute(0, 2, 3, 1) + 1) / 2, 0, 1)[idx].cpu().detach().numpy())
                axes[row_idx * 4 + 0, col_idx * (c.steps + 1) + k].title.set_text(f'Render @ Step {k}')

                # Sprites
                if k > 0:
                    axes[row_idx * 4 + 1, col_idx * (c.steps + 1) + k].imshow((sprites)[idx, ..., k - 1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 4 + 1, col_idx * (c.steps + 1) + k].title.set_text(f'Sprite @ Step {k}')
                    
                    center_x, center_y = centers[idx, k - 1, 0:2].detach().cpu().numpy()
                    center = ((center_y + 1) / 2 * size, (center_x + 1) / 2 * size)
                    axes[row_idx * 4 + 1, col_idx * (c.steps + 1) + k].add_patch(Circle(center, 1, color='red'))

                # Masks
                if k > 0:
                    axes[row_idx * 4 + 2, col_idx * (c.steps + 1) + k].imshow((masks)[idx, ..., k-1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 4 + 2, col_idx * (c.steps + 1) + k].title.set_text(f'Mask @ Step {k}')
                else:
                    axes[row_idx * 4 + 2, col_idx * (c.steps + 1) + k].imshow((masks)[idx, ..., -1].cpu().detach().numpy(), vmin=0, vmax=1)
                    axes[row_idx * 4 + 2, col_idx * (c.steps + 1) + k].title.set_text(f'Mask @ Step {k}')

                # Texture

                if k > 0:
                    axes[row_idx * 4 + 3, col_idx * (c.steps + 1) + k].imshow(torch.clamp((textures[idx, ..., k-1, :3] + 1) / 2, 0, 1).cpu().detach().numpy())
                    axes[row_idx * 4 + 3, col_idx * (c.steps + 1) + k].title.set_text(f'Texture @ Step {k}')
                else:
                    axes[row_idx * 4 + 3, col_idx * (c.steps + 1) + k].imshow(torch.clamp((textures[idx, ..., -1, :3] +1) / 2, 0, 1).cpu().detach().numpy())
                    axes[row_idx * 4 + 3, col_idx * (c.steps + 1) + k].title.set_text(f'Texture @ Step {k}')

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

    import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, flush=True)

def training_loop(c):
    torch.backends.cudnn.benchmark = True               # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.


    print(torch.cuda.nccl.version(), flush=True)

    # Load training data
    dataloader = get_dataloader(c)

    # Instantiate generator and discriminator
    generator = MultiShapeStyleLinearApproach(**c.G_args).to(c.device)
    my_regularizer = MyRegularizer(c.G_args.shape_style_dim, generator.to_rgb.num_layers, final_activation=None).to(c.device)
    discriminator = StyleGANDiscriminator(c.G_args.output_size, in_channels=3, channel_multiplier=1).to(c.device)
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
            my_regularizer.load_state_dict(loaded_dict['my_regularizer'])
            discriminator.load_state_dict(loaded_dict['discriminator'])

            c.ema_rampup = None # Disable ema rampup
    
    # Distribute across GPUs.
    if c.rank == 0:
        print(f'Distributing across {c.world_size} GPUs...')
    if c.world_size > 1:
        for module in [generator, discriminator, my_regularizer, generator_ema]:
            if module is not None:
                for param in params_and_buffers(module):
                    dist.broadcast(param, src=0)

    # Setup loss and optimizers
    loss = MyLoss(generator, my_regularizer, discriminator, c)

    g_mb_ratio = (c.g_reg_every) / (c.g_reg_every + 1)
    optimizerG = torch.optim.Adam(
        [
            {'params': generator.parameters()},
            {'params': my_regularizer.parameters()},
        ],
        lr=c.generator_lr * g_mb_ratio, betas=(0.0, 0.99 ** g_mb_ratio), eps=1e-8
    )

    d_mb_ratio = (c.d_reg_every) / (c.d_reg_every + 1)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=c.discriminator_lr * d_mb_ratio, betas=(0.0, 0.99 ** d_mb_ratio), eps=1e-8)

    # Setup logging
    if c.rank == 0:
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(project="my-test-project", entity="amsabour", resume="allow", dir=c.save_dir) # , id=c.job_id)
        wandb.config.update(c, allow_val_change=True)
    

    tick_start_global_step = c.global_step
    plot_start_global_step = c.global_step
    snap_start_global_step = c.global_step
    noise = torch.randn(64, c.G_args.hidden_dim).to(c.device)

    if c.global_step < c.progressive_start_unlock_kimg * 1000:
        generator.assembler.shape_library.requires_grad_(False)
        # generator.assembler.texture_library.requires_grad_(False)
        

    while c.global_step < c.kimg * 1000:
        for i, (x, _) in enumerate(dataloader):
            c.global_step += c.batch_size

            start_fetch_data = time.time()
            x = x.to(c.device)
            end_fetch_data = time.time()
            log_wandb({'Timings/data_fetch': end_fetch_data - start_fetch_data}, c)

            # G Step
            if i % 1 == 0:
                generator.requires_grad_(True)
                if c.global_step < c.progressive_start_unlock_kimg * 1000:
                    generator.assembler.shape_library.requires_grad_(False)
                    # generator.assembler.texture_library.requires_grad_(False)
                
                my_regularizer.requires_grad_(True)
                discriminator.requires_grad_(False)

                optimizerG.zero_grad(set_to_none=True)
                start_G = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0): 
                    loss.accumulate_gradients('G', xx, gain=1 / c.num_batch_splits, log=(i % 1 == 0))
                end_G = time.time()
                accumulate_gradients([generator, my_regularizer], c)
                log_wandb({'Gradients/G_grad_norm': get_grad_norm([generator])}, c)
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
                if c.global_step < c.progressive_start_unlock_kimg * 1000:
                    generator.assembler.shape_library.requires_grad_(False)
                    # generator.assembler.texture_library.requires_grad_(False)

                my_regularizer.requires_grad_(False)
                discriminator.requires_grad_(False)
                
                optimizerG.zero_grad(set_to_none=True)
                start_G_reg = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0): 
                    loss.accumulate_gradients('G_reg', xx, gain=c.g_reg_every / c.num_batch_splits, log=True)
                end_G_reg = time.time()
                accumulate_gradients([generator], c)
                log_wandb({'Gradients/G_reg_grad_norm': get_grad_norm([generator])}, c)
                end_G_reg_acc = time.time()
                optimizerG.step()
                end_G_reg_opt = time.time()

                log_wandb({
                    'Timings/G_reg_forward_backward': end_G_reg - start_G_reg,
                    'Timings/G_reg_accumulate': end_G_reg_acc - end_G_reg,
                    'Timings/G_reg_opt': end_G_reg_opt - end_G_reg_acc
                }, c)

            # D Step
            if i % 1 == 0:
                generator.requires_grad_(False)
                my_regularizer.requires_grad_(False)
                discriminator.requires_grad_(True)

                optimizerD.zero_grad(set_to_none=True)
                start_D = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0): 
                    loss.accumulate_gradients('D', xx, gain=1 / c.num_batch_splits, log=(i % 1 == 0))
                end_D = time.time()
                accumulate_gradients([discriminator], c)
                log_wandb({'Gradients/D_grad_norm': get_grad_norm([discriminator])}, c)
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

                optimizerD.zero_grad()
                start_D_reg = time.time()
                for xx in torch.chunk(x, c.num_batch_splits, dim=0):
                    loss.accumulate_gradients('D_reg', xx, gain=c.d_reg_every / c.num_batch_splits, log=True)
                end_D_reg = time.time()
                accumulate_gradients([discriminator], c)
                log_wandb({'Gradients/D_reg_grad_norm': get_grad_norm([discriminator])}, c)
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
            if c.global_step > c.progressive_start_unlock_kimg * 1000:
                generator.assembler.shape_library.requires_grad_(True)
                # generator.assembler.texture_library.requires_grad_(True)

                generator.assembler.shape_library.step(c.batch_size / (c.shape_progressive_unlock_kimg * 1000))
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
            log_wandb({'Timings/ema_update': end_ema - start_ema} ,c)
            
            if c.rank == 0:
                wandb.log({"commited": True}, step=c.global_step)
                    
                # Plot samples and reconstructions
                if (c.global_step - plot_start_global_step) > c.plot * 1000 or c.global_step == c.batch_size:
                    plot_start_global_step = c.global_step
                    plot_training_progress(generator_ema, noise, c)
                    
                # Save checkpoint
                if (c.global_step - snap_start_global_step) > c.snap * 1000:
                    snap_start_global_step = c.global_step
                    save_checkpoint(generator, generator_ema, my_regularizer, discriminator, c.global_step, c.checkpoint_path)

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
@click.option('--complexity-weight',  help='Sprite complexity regularization weight', metavar='FLOAT',  type=click.FloatRange(min=0), default=0, show_default=True)


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


@click.option('--use-noise',                            help='Use per-pixel noise injections', metavar='BOOL',             type=bool, default=False, show_default=True)

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
def main(**kwargs):

    ################ Setup Config ################  
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments.
    c = dnnlib.EasyDict()  # Main config dict.
    
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
    c.complexity_weight = opts.complexity_weight

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

    c.steps = opts.steps
    c.shape_progressive_unlock_kimg = opts.shape_progressive_unlock_kimg
    c.texture_progressive_unlock_kimg = opts.texture_progressive_unlock_kimg
    c.progressive_start_unlock_kimg = opts.progressive_start_unlock_kimg
    c.ema_kimg = opts.ema_kimg
    c.ema_rampup = opts.ema_rampup
    c.use_noise = opts.use_noise

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