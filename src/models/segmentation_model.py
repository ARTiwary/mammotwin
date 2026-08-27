"""
Phase 8: standard U-Net for lesion segmentation.

Written directly in PyTorch rather than pulling in MONAI's UNet, following
the lesson from Phase 7 -- fewer dependencies means fewer ways for an
environment-specific issue (like the scipy DLL block we hit) to derail
training. U-Net's architecture is simple enough that a from-scratch
implementation is standard practice and easy to describe exactly in a
methods section.

Single-channel input (preprocessed mammogram, already normalized to [0,1]
by Phase 4), single-channel output (per-pixel lesion probability via
sigmoid, paired with Dice/BCE loss during training).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2 -- the basic U-Net building block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # raw logits -- apply sigmoid outside for probabilities


def build_segmentation_model(config: dict) -> nn.Module:
    seg_cfg = config.get("segmentation", {})
    base_filters = seg_cfg.get("base_filters", 32)
    return UNet(in_channels=1, out_channels=1, base_filters=base_filters)
