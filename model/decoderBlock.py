import torch
from torch import nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md

from attentionModules import MultiScaleAttention


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        if skip_channels > 0:
            self.multiAttention = MultiScaleAttention(skip_channels)
        else:
            self.multiAttention = nn.Identity()

    def forward(self, x, skip=None, i=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if i is not None and i == 2:
                skip = self.multiAttention(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
