"""Training losses."""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
from .discriminator import DEFAULT_STFT_SIZES


def reconstruction_loss(x_real, x_hat, step, initial_weight=1.0, decay_steps=50000.0):
    weight = initial_weight * torch.exp(
        torch.tensor(-step / decay_steps, device=x_real.device, dtype=x_real.dtype)).item()
    return weight * F.l1_loss(x_hat, x_real), weight


def stft_magnitude_loss(x_real, x_hat, fft_sizes=DEFAULT_STFT_SIZES):
    total = x_real.new_zeros(())
    xr, xh = x_real.squeeze(1), x_hat.squeeze(1)
    for size in fft_sizes:
        w = torch.hann_window(size, device=x_real.device)
        hop = max(size // 4, 1)
        sr = torch.stft(xr, size, hop, size, w, return_complex=True, normalized=False, onesided=True)
        sf = torch.stft(xh, size, hop, size, w, return_complex=True, normalized=False, onesided=True)
        total = total + F.l1_loss(sf.abs(), sr.abs())
    return total / max(len(fft_sizes), 1)


def feature_matching_loss(fmaps_real, fmaps_fake):
    total = torch.tensor(0.0, device=fmaps_real[0][0].device)
    n = 0
    for fr_list, ff_list in zip(fmaps_real, fmaps_fake):
        for fr, ff in zip(fr_list, ff_list):
            total = total + F.l1_loss(ff, fr.detach())
            n += 1
    return total / max(n, 1)


def discriminator_loss(logits_real, logits_fake):
    loss = logits_real[0].new_zeros(())
    for lr, lf in zip(logits_real, logits_fake):
        loss = loss + F.relu(1.0 - lr).mean() + F.relu(1.0 + lf).mean()
    return loss / max(len(logits_real), 1)


def generator_adversarial_loss(logits_fake):
    loss = logits_fake[0].new_zeros(())
    for lf in logits_fake:
        loss = loss - lf.mean()
    return loss / max(len(logits_fake), 1)
