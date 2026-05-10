"""Single-codebook VQ quantizer."""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=8192, dim=256, commitment_cost=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def lookup(self, indices):
        B, T = indices.shape
        z_q = self.codebook(indices.reshape(-1)).view(B, T, self.dim)
        return z_q.permute(0, 2, 1).contiguous()

    def forward(self, z):
        B, D, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(B * T, D)
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2.0 * z_flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1).unsqueeze(0))
        indices = dist.argmin(dim=1)
        z_q = self.lookup(indices.view(B, T))
        cb_loss = F.mse_loss(z_q, z.detach())
        commit_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return z_q, indices.view(B, T), cb_loss + commit_loss


class SingleCodebookQuantizer(nn.Module):
    def __init__(self, latent_dim=256, codebook_size=8192, commitment_cost=0.25):
        super().__init__()
        self.vq = VectorQuantizer(codebook_size, latent_dim, commitment_cost)
        self.latent_dim = latent_dim

    def forward(self, z):
        return self.vq(z)
