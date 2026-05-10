"""Voxtral Encoder - causal conv-transformer."""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand(value, n, name):
    if isinstance(value, int): return (value,) * n
    v = tuple(value)
    assert len(v) == n, f"{name} length mismatch"
    return v


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.ps = patch_size

    def forward(self, x):
        B, C, T = x.shape
        pad = (-T) % self.ps
        if pad: x = F.pad(x, (0, pad))
        frames = x.shape[-1] // self.ps
        return x.view(B, 1, frames, self.ps).squeeze(1).transpose(1, 2).contiguous()


class ResidualCausalBlock(nn.Module):
    def __init__(self, ch, kernel_size=7, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(), CausalConv1d(ch, ch, kernel_size, dilation=dilation),
            nn.ELU(), CausalConv1d(ch, ch, 1))

    def forward(self, x): return x + self.net(x)


class LayerScale(nn.Module):
    def __init__(self, dim, init=0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x): return x * self.scale


def _alibi_slopes(n_heads, device, dtype):
    base = torch.linspace(0, 1, n_heads, device=device, dtype=dtype)
    return torch.pow(torch.tensor(2.0, device=device, dtype=dtype), -(8.0 * base))


class CausalSlidingWindowAttention(nn.Module):
    def __init__(self, dim, n_heads, window_size, eps=1e-6):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.window_size, self.eps = window_size, eps
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)
        attn = q @ k.transpose(-2, -1)
        pos = torch.arange(T, device=x.device)
        rel = pos[None, :] - pos[:, None]
        mask = rel < 0
        if self.window_size > 0: mask = mask | (rel >= self.window_size)
        slopes = _alibi_slopes(self.n_heads, x.device, x.dtype).view(1, self.n_heads, 1, 1)
        alibi = -rel.abs().to(x.dtype).view(1, 1, T, T) * slopes
        attn = attn + alibi.masked_fill(mask.view(1, 1, T, T), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSlidingWindowAttention(dim, n_heads, window_size)
        self.scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.scale2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.ffn(self.norm2(x)))
        return x


class SlidingWindowTransformer(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, n_layers, window_size):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(dim, n_heads, ffn_dim, window_size) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers: x = layer(x)
        return self.norm(x).transpose(1, 2)


class CausalDownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, kernel_size, n_residual, dilations):
        super().__init__()
        self.residuals = nn.Sequential(*[ResidualCausalBlock(in_ch, 7, d) for d in dilations])
        self.downsample = nn.Sequential(nn.ELU(), CausalConv1d(in_ch, out_ch, kernel_size, stride=stride))

    def forward(self, x): return self.downsample(self.residuals(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, kernel_size, n_residual, dilations,
                 n_transformer_layers, n_heads, ffn_dim, window_size):
        super().__init__()
        self.transformer = SlidingWindowTransformer(in_ch, n_heads, ffn_dim, n_transformer_layers, window_size)
        self.cnn = CausalDownsampleBlock(in_ch, out_ch, stride, kernel_size, n_residual, dilations)

    def forward(self, x): return self.cnn(self.transformer(x))


class VoxtralEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=512, latent_dim=256, patch_stride=240,
                 patch_kernel_size=7, block_strides=(2,2,2,1), block_kernel_sizes=(4,4,4,3),
                 n_residual=3, dilations=(1,3,9), n_transformer_layers=2, n_heads=8,
                 ffn_dim=2048, window_size=(16,8,4,2)):
        super().__init__()
        n_blocks = len(block_strides)
        block_kernel_sizes = _expand(block_kernel_sizes, n_blocks, "k")
        window_sizes = _expand(window_size, n_blocks, "w")
        tl = _expand(n_transformer_layers, n_blocks, "tl")
        self.patchify = Patchify(patch_stride)
        self.patch_proj = nn.Sequential(CausalConv1d(patch_stride, hidden_dim, patch_kernel_size), nn.ELU())
        dims = (hidden_dim, hidden_dim, hidden_dim, latent_dim)
        self.blocks = nn.ModuleList()
        in_d = hidden_dim
        for od, s, ks, nl, ws in zip(dims, block_strides, block_kernel_sizes, tl, window_sizes):
            self.blocks.append(EncoderBlock(in_d, od, s, ks, n_residual, dilations, nl, n_heads, ffn_dim, ws))
            in_d = od

    def forward(self, x):
        x = self.patch_proj(self.patchify(x))
        for block in self.blocks: x = block(x)
        return x
