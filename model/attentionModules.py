from torch import nn
import torch
from convolutionBase import Convolution
import torch.nn.functional as F


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = Convolution(in_channels, in_channels, 3)
        self.conv2 = nn.Sequential(
            Convolution(in_channels, in_channels / 4, 3),
            Convolution(in_channels / 4, in_channels / 4, 1)
        )
        self.conv3 = Convolution(in_channels, in_channels / 4, 1)
        self.conv1 = nn.Sequential(
            Convolution(in_channels, in_channels / 2, 3),
            Convolution(in_channels / 2, in_channels / 2, 3, rate=2),
            Convolution(in_channels / 2, in_channels / 2, 1)
        )
        self.comb_conv = Convolution(in_channels, in_channels, 1)
        self.final = Convolution(2 * in_channels, in_channels, 3, 2)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_comb = torch.cat((x1, x2, x3), dim=1)
        x_n = self.comb_conv(x_comb)
        x_new = torch.cat((x, x_n), dim=1)
        out = self.final(x_new)
        return out


class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.query, self.key, self.value = [self._conv(n_channels, c) for c in
                                            (n_channels // 8, n_channels // 8, n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    @staticmethod
    def _conv(n_in, n_out):
        return nn.Conv1d(n_in, n_out, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
