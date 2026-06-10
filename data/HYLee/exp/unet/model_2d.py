# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock2D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvBlock2D(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """2D U-Net for mel-spectrogram segmentation.

    Input
        x: [B, in_channels, n_mels, n_frames]

    Output
        logits: [B, out_channels, n_frames]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        encoder_channels: Tuple[int, int, int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = (
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
            )

        c1, c2, c3, c4 = encoder_channels
        bottleneck_ch = c4 * 2

        self.enc1 = ConvBlock2D(in_channels, c1, dropout=dropout)
        self.enc2 = ConvBlock2D(c1, c2, dropout=dropout)
        self.enc3 = ConvBlock2D(c2, c3, dropout=dropout)
        self.enc4 = ConvBlock2D(c3, c4, dropout=dropout)
        self.bottleneck = ConvBlock2D(c4, bottleneck_ch, dropout=dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dec4 = UpBlock2D(bottleneck_ch, c4, c4, dropout=dropout)
        self.dec3 = UpBlock2D(c4, c3, c3, dropout=dropout)
        self.dec2 = UpBlock2D(c3, c2, c2, dropout=dropout)
        self.dec1 = UpBlock2D(c2, c1, c1, dropout=dropout)

        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, C, F, T], got {tuple(x.shape)}")

        input_frames = x.shape[-1]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        logits_2d = self.final_conv(d1)
        logits = logits_2d.mean(dim=2)

        if logits.shape[-1] != input_frames:
            logits = F.interpolate(logits, size=input_frames, mode="linear", align_corners=False)
        return logits


if __name__ == "__main__":
    model = UNet2D(in_channels=3, out_channels=5, base_channels=32)
    x = torch.randn(2, 3, 128, 1001)
    y = model(x)
    print("input", tuple(x.shape))
    print("output", tuple(y.shape))
    print("finite", torch.isfinite(y).all().item())
