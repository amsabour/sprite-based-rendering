# https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .op import FusedLeakyReLU, conv2d_gradfix

from .stylegan import PixelNorm, EqualLinear, ConstantInput, Blur, NoiseInjection, Upsample


class SpatialToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], c_out=3):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = SpatialModulatedConv2d(in_channel, c_out, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, c_out, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class SpatialStyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = SpatialModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )
        
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class SpatialModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.style_dim = style_dim
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style, spatial_style=True):
        batch, in_channel, height, width = input.shape
        assert height == width
        size = height

        if spatial_style:
            if isinstance(style, dict):
                masks = style['masks'] # [batch_size, H, W, num_shapes+1]
                masks = F.interpolate(rearrange(masks, 'b H W s -> b s H W'), (size, size), mode='bilinear', align_corners=False)

                textures = style['textures'] # [batch_size, H, W, num_shapes+1, shape_style_dim]
                num_textures = textures.shape[-2]
                texture_dim = textures.shape[-1]
                textures = F.interpolate(rearrange(textures, 'b H W s d -> b (s d) H W'), (size, size), mode='bilinear', align_corners=False)
                textures = rearrange(textures, 'b (s d) H W -> b (H W s) d', s=num_textures, d=texture_dim)
                textures = rearrange(self.modulation(textures), 'b (H W s) d -> b d s H W', H=size, W=size, s=num_textures)
                
                style = (masks[:, None] * textures).sum(dim=2)
            else:
                # Interpolate 
                style = F.interpolate(style, (size, size), mode='bilinear', align_corners=False)

                # style is [N, D_style, H, W] --> [N, in_channels, H, W]
                style = self.modulation(style.permute(0, 2, 3, 1).flatten(end_dim=-2)).view(style.shape[0], style.shape[2], style.shape[3], -1).permute(0, 3, 1, 2)

            if self.demodulate:
                style = style * torch.rsqrt(style.pow(2).mean([1], keepdim=True) + 1e-8)
        else:
            style = self.modulation(style).reshape(batch, in_channel, 1, 1)
        
        weight = self.scale * self.weight.squeeze(0)
        input = input * style

        if self.demodulate:
            if spatial_style:
                demod = torch.rsqrt(weight.unsqueeze(0).pow(2).sum([2, 3, 4]) + 1e-8)
                weight = weight * demod.view(self.out_channel, 1, 1, 1)
            else:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

        if self.upsample:
            weight = weight.transpose(0, 1)
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2
            )
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

        else:
            out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

        if self.demodulate and not spatial_style:
            out = out * dcoefs.view(batch, -1, 1, 1)

        return out
    
