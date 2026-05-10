"""Multi-resolution STFT discriminator."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple

DEFAULT_STFT_SIZES = (2296, 1418, 876, 542, 334, 206, 126, 76)

def _stft(x, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=x.device)
    s = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, return_complex=True, normalized=False, onesided=True)
    return torch.stack([s.real, s.imag], dim=1)


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft, hop_length=None, win_length=None, channels=256, n_layers=4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.win_length = win_length or n_fft
        in_ch = 2
        layers = []
        for i in range(n_layers):
            out_ch = min(channels * (2 ** i), 1024)
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.1, inplace=True)))
            in_ch = out_ch
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(0.1, inplace=True)))
        self.convs = nn.ModuleList(layers)
        self.conv_post = nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        x = x.squeeze(1)
        spec = _stft(x, self.n_fft, self.hop_length, self.win_length)
        fmaps = []
        h = spec
        for conv in self.convs:
            h = conv(h)
            fmaps.append(h)
        return self.conv_post(h), fmaps


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, channels=256, n_layers=4):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft=s, hop_length=max(s//4,1), win_length=s,
                              channels=channels, n_layers=n_layers)
            for s in DEFAULT_STFT_SIZES
        ])

    def forward(self, x):
        return [disc(x) for disc in self.discriminators]
