"""Single-codebook Voxtral Codec model."""
from __future__ import annotations
from typing import Dict, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import VoxtralEncoder
from .decoder import VoxtralDecoder
from .quantizer import SingleCodebookQuantizer


class SingleCodebookCodec(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=512, latent_dim=256, patch_stride=240,
                 encoder_strides=(2,2,2,1), decoder_strides=(1,2,2,2),
                 encoder_kernel_sizes=(4,4,4,3), decoder_kernel_sizes=(3,4,4,4),
                 patch_kernel_size=7, n_residual=3, dilations=(1,3,9),
                 n_transformer_layers=2, n_heads=8, ffn_dim=2048,
                 window_size=(16,8,4,2), codebook_size=8192,
                 commitment_cost=0.25, sample_rate=24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.patch_stride = patch_stride
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        stride = 1
        for s in encoder_strides: stride *= s
        self.total_stride = patch_stride * stride
        self.frame_rate = sample_rate / self.total_stride

        self.encoder = VoxtralEncoder(
            in_channels=in_channels, hidden_dim=hidden_dim, latent_dim=latent_dim,
            patch_stride=patch_stride, patch_kernel_size=patch_kernel_size,
            block_strides=encoder_strides, block_kernel_sizes=encoder_kernel_sizes,
            n_residual=n_residual, dilations=dilations,
            n_transformer_layers=n_transformer_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, window_size=window_size)

        self.quantizer = SingleCodebookQuantizer(latent_dim, codebook_size, commitment_cost)

        self.decoder = VoxtralDecoder(
            out_channels=in_channels, hidden_dim=hidden_dim, latent_dim=latent_dim,
            patch_stride=patch_stride, patch_kernel_size=patch_kernel_size,
            block_strides=decoder_strides, block_kernel_sizes=decoder_kernel_sizes,
            n_residual=n_residual, dilations=dilations,
            n_transformer_layers=n_transformer_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, window_size=tuple(reversed(window_size)))

    def forward(self, x):
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        x_hat = self.decoder(z_q)
        Ti, To = x.shape[-1], x_hat.shape[-1]
        if To > Ti: x_hat = x_hat[..., :Ti]
        elif To < Ti: x_hat = F.pad(x_hat, (0, Ti - To))
        return {"x_hat": x_hat, "z": z, "z_q": z_q, "indices": indices, "vq_loss": vq_loss}

    def encode(self, x):
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices

    def decode_from_codes(self, indices):
        z_q = self.quantizer.vq.lookup(indices)
        return self.decoder(z_q)

    def info(self):
        c = lambda m: sum(p.numel() for p in m.parameters())
        return (f"SingleCodebookCodec | {self.frame_rate:.1f}Hz | "
                f"dim={self.latent_dim} | CB={self.codebook_size} | "
                f"Enc={c(self.encoder)/1e6:.1f}M Dec={c(self.decoder)/1e6:.1f}M "
                f"Total={c(self)/1e6:.1f}M")
