import torch
from torch import nn
import torch.nn.functional as F
from convblock import *

# Taken and modified from https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    # Creating few conv blocks. One with kernel 3, second with kernel 1. With residual skip connection
    def __init__(self, ch, nblocks=1, shortcut=True, dropblock=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBlock(ch, ch, 1, 1, 'mish', dropblock=dropblock))
            resblock_one.append(ConvBlock(ch, ch, 3, 1, 'mish', dropblock=dropblock))
            self.module_list.append(resblock_one)

        if dropblock:
            self.use_dropblock = True
            self.dropblock = DropBlock2D()
        else:
            self.use_dropblock = False

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
            if self.use_dropblock:
                x = self.dropblock(x)

        return x

class DownSampleFirst(nn.Module):
    """
    This is first downsample of the backbone model.
    It differs from the other stages, so it is written as another Module
    Args:
        in_channels (int): Amount of channels to input, if you use RGB, it should be 3
    """
    def __init__(self, in_channels=3, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False):
        super().__init__()

        self.c1 = ConvBlock(in_channels, 32, 3, 1, "mish", dropblock=dropblock)
        self.c2 = ConvBlock(32, 64, 3, 2, "mish", dropblock=dropblock)
        self.c3 = ConvBlock(64, 64, 1, 1, "mish", dropblock=dropblock)
        self.c4 = ConvBlock(64, 32, 1, 1, "mish", dropblock=dropblock)
        self.c5 = ConvBlock(32, 64, 3, 1, "mish", dropblock=dropblock)
        self.c6 = ConvBlock(64, 64, 1, 1, "mish", dropblock=dropblock)

        # CSP Layer
        self.dense_c3_c6 = ConvBlock(64, 64, 1, 1, "mish", dropblock=dropblock)

        self.c7 = ConvBlock(128, 64, 1, 1, "mish", dropblock=dropblock)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x5 = x5 + x3    # Residual block
        x6 = self.c6(x5)
        xd6 = self.dense_c3_c6(x2)  # CSP
        x6 = torch.cat([x6, xd6], dim=1)
        x7 = self.c7(x6)
        return x7


class DownSampleBlock(nn.Module):
    def __init__(self, in_c, out_c, nblocks=2, dropblock=True):
        super().__init__()

        self.c1 = ConvBlock(in_c, out_c, 3, 2, "mish", dropblock=dropblock)
        self.c2 = ConvBlock(out_c, in_c, 1, 1, "mish", dropblock=dropblock)
        self.r3 = ResBlock(in_c, nblocks=nblocks, dropblock=dropblock)
        self.c4 = ConvBlock(in_c, in_c, 1, 1, "mish", dropblock=dropblock)

        # CSP Layer
        self.dense_c2_c4 = ConvBlock(out_c, in_c, 1, 1, "mish", dropblock=dropblock)

        self.c5 = ConvBlock(out_c, out_c, 1, 1, "mish", dropblock=dropblock)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.r3(x2)
        x4 = self.c4(x3)
        xd4 = self.dense_c2_c4(x1)  # CSP
        x4 = torch.cat([x4, xd4], dim=1)
        x5 = self.c5(x4)

        return x5

class Backbone(nn.Module):
    def __init__(self, in_channels, dropblock=True):
        super().__init__()

        self.d1 = DownSampleFirst(in_channels=in_channels, dropblock=dropblock)
        self.d2 = DownSampleBlock(64, 128, nblocks=2, dropblock=dropblock)
        self.d3 = DownSampleBlock(128, 256, nblocks=8, dropblock=dropblock)
        self.d4 = DownSampleBlock(256, 512, nblocks=8, dropblock=dropblock)
        self.d5 = DownSampleBlock(512, 1024, nblocks=4, dropblock=dropblock)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        return (x5, x4, x3)