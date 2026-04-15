import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, downsampling=None):
        super().__init__()
        if downsampling not in ["maxpool", "avgpool"]:
            raise ValueError("downsampling must be 'maxpool' or 'avgpool'")
        self.downsampling = downsampling

        # Stage 1 (32x32)
        self.c1 = ConvBlock(3, 32)
        self.c2 = ConvBlock(32, 32)
        self.ds1 = self._make_downsample(32, 64)

        # Stage 2 (16x16)
        self.c3 = ConvBlock(64, 64)
        self.c4 = ConvBlock(64, 64)
        self.ds2 = self._make_downsample(64, 128)

        # Stage 3 (8x8)
        self.c5 = ConvBlock(128, 128)
        self.c6 = ConvBlock(128, 128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_downsample(self, in_ch, out_ch):
        if self.downsampling == "maxpool":
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBlock(in_ch, out_ch, k=3, s=1, p=1),
            )
        elif self.downsampling == "avgpool":
            return nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                ConvBlock(in_ch, out_ch, k=3, s=1, p=1),
            )
        raise ValueError("downsampling must be 'maxpool' or 'avgpool'")

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.ds1(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.ds2(x)

        x = self.c5(x)
        x = self.c6(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
