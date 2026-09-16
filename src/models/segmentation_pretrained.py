"""
Phase 8 (accuracy-improvement follow-up): U-Net with a pretrained ImageNet
ResNet encoder, as an alternative to the from-scratch UNet in
segmentation_model.py.

Why: the from-scratch U-Net has to learn every feature -- edges, textures,
shapes -- purely from ~2,400 training images, which is a small dataset for
learning good low-level visual features from nothing. A pretrained ResNet
encoder already knows a huge library of general-purpose visual features
(edges, blobs, textures) from ImageNet, and only needs to be fine-tuned to
recognize mammographic lesion boundaries specifically -- typically the
single highest-leverage change for small-data segmentation tasks in the
literature.

This is a genuine architecture swap, not a cosmetic one: the encoder path
(everything up to the bottleneck) is a real pretrained torchvision ResNet;
the decoder (upsampling + skip connections back to full resolution) is
still written by hand here, same spirit as the rest of this project's
"a from-scratch implementation is standard practice and easy to describe
in a methods section" philosophy in segmentation_model.py.

Input handling: ResNet expects 3-channel input. Rather than touching the
pretrained conv1 weights, the 1-channel preprocessed mammogram is simply
repeated to 3 channels before the encoder -- this keeps 100% of the
pretrained weights untouched (an alternative approach, averaging conv1's
3-channel weights down to 1 channel, saves a little compute but is not
necessary here and adds a subtlety not worth the complexity).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ConvBlock(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2 -- same decoder building block as
    the from-scratch UNet, for a fair, consistent comparison between the
    two architectures."""
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


class UpBlock(nn.Module):
    """Upsample, concatenate the matching encoder skip connection, then a
    ConvBlock -- standard U-Net decoder stage."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Guard against off-by-one size mismatches from odd input
        # dimensions propagating through repeated stride-2 downsampling.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PretrainedUNet(nn.Module):
    """
    U-Net-style decoder over a pretrained torchvision ResNet encoder.

    Skip connections are taken at each ResNet stage boundary:
        stem   (after conv1+bn1+relu, before maxpool): 64ch  @ 1/2 resolution
        layer1: 64ch  @ 1/4
        layer2: 128ch @ 1/8
        layer3: 256ch @ 1/16
        layer4: 512ch @ 1/32  (bottleneck, resnet18/34)
    """

    _ENCODER_CHANNELS = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
    }

    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True, out_channels: int = 1):
        super().__init__()
        if encoder_name not in self._ENCODER_CHANNELS:
            raise ValueError(f"Unsupported encoder_name: {encoder_name!r}")

        if encoder_name == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tv_models.resnet18(weights=weights)
        else:
            weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tv_models.resnet34(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # -> 64ch @ 1/2
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # -> 64ch  @ 1/4
        self.layer2 = resnet.layer2  # -> 128ch @ 1/8
        self.layer3 = resnet.layer3  # -> 256ch @ 1/16
        self.layer4 = resnet.layer4  # -> 512ch @ 1/32 (bottleneck)

        c = self._ENCODER_CHANNELS[encoder_name]  # [stem, layer1, layer2, layer3, layer4]

        self.up4 = UpBlock(c[4], c[3], 256)   # bottleneck -> 1/16, fuse layer3
        self.up3 = UpBlock(256, c[2], 128)    # -> 1/8, fuse layer2
        self.up2 = UpBlock(128, c[1], 64)     # -> 1/4, fuse layer1
        self.up1 = UpBlock(64, c[0], 32)      # -> 1/2, fuse stem
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # -> full resolution
        self.final_conv = ConvBlock(16, 16)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale -> 3ch for the pretrained encoder, weights untouched

        s0 = self.stem(x)              # 64ch  @ 1/2
        p0 = self.maxpool(s0)
        s1 = self.layer1(p0)           # 64ch  @ 1/4
        s2 = self.layer2(s1)           # 128ch @ 1/8
        s3 = self.layer3(s2)           # 256ch @ 1/16
        s4 = self.layer4(s3)           # 512ch @ 1/32 (bottleneck)

        d4 = self.up4(s4, s3)
        d3 = self.up3(d4, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)
        d0 = self.final_up(d1)
        if d0.shape[-2:] != x.shape[-2:]:
            d0 = nn.functional.interpolate(d0, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.final_conv(d0)
        return self.out_conv(d0)  # raw logits -- apply sigmoid outside, same convention as UNet
