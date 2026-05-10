"""Voxtral Decoder - causal upsampling conv-transformer."""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
from .encoder import CausalConv1d, ResidualCausalBlock, SlidingWindowTransformer, _expand


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, bias=True):
        super().__init__()
        self.trim = max(kernel_size - stride, 0)
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        x = self.deconv(x)
        if self.trim: x = x[..., :-self.trim]
        return x


class CausalUpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, kernel_size, n_residual=3, dilations=(1,3,9)):
        super().__init__()
        self.upsample = nn.Sequential(nn.ELU(), CausalConvTranspose1d(in_ch, out_ch, kernel_size, stride))
        self.residuals = nn.Sequential(*[ResidualCausalBlock(out_ch, kernel_size, d) for d in dilations])

    def forward(self, x): return self.residuals(self.upsample(x))


class DecoderBlock(nn.Module):
    def __init__(self, channels, stride, kernel_size, n_residual, dilations,
                 n_transformer_layers, n_heads, ffn_dim, window_size):
        super().__init__()
        self.cnn = CausalUpsampleBlock(channels, channels, stride, kernel_size, n_residual, dilations)
        self.transformer = SlidingWindowTransformer(channels, n_heads, ffn_dim, n_transformer_layers, window_size)

    def forward(self, x): return self.transformer(self.cnn(x))


class VoxtralDecoder(nn.Module):
    def __init__(self, out_channels=1, hidden_dim=512, latent_dim=256, patch_stride=240,
                 patch_kernel_size=7, block_strides=(1,2,2,2), block_kernel_sizes=(3,4,4,4),
                 n_residual=3, dilations=(1,3,9), n_transformer_layers=2, n_heads=8,
                 ffn_dim=2048, window_size=(2,4,8,16)):
        super().__init__()
        n_blocks = len(block_strides)
        block_kernel_sizes = _expand(block_kernel_sizes, n_blocks, "k")
        window_sizes = _expand(window_size, n_blocks, "w")
        tl = _expand(n_transformer_layers, n_blocks, "tl")
        self.input_proj = nn.Sequential(CausalConv1d(latent_dim, hidden_dim, 1), nn.ELU())
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden_dim, s, ks, n_residual, dilations, nl, n_heads, ffn_dim, ws)
            for s, ks, nl, ws in zip(block_strides, block_kernel_sizes, tl, window_sizes)
        ])
        self.output_proj = nn.Sequential(nn.ELU(), CausalConv1d(hidden_dim, patch_stride, patch_kernel_size), nn.Tanh())
        self.patch_stride = patch_stride

    def forward(self, z):
        x = self.input_proj(z)
        for block in self.blocks: x = block(x)
        x = self.output_proj(x)
        B, P, Fr = x.shape
        return x.transpose(1, 2).reshape(B, 1, Fr * P)
