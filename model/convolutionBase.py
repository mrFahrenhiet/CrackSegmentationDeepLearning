from torch import nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, rate=1):
        super().__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels),
                      out_channels=int(out_channels),
                      kernel_size=(k_size, k_size),
                      dilation=(rate, rate),
                      padding='same'),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.convlayer(x)
