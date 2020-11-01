import torch
from torch import nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * torch.tanh(F.softplus(x))


class DropBlock2D(nn.Module):

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        # print("Before: ", torch.isnan(input).sum())
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma).to(device=input.device)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(device=input.device,
                                                                                             dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        # print("After: ", torch.isnan(input * mask * mask.numel() /mask.sum()).sum())
        return input * mask * mask.numel() / mask.sum()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, dropblock=False, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))

        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!!") 
        if dropblock:
        	self.conv.append(DropBlock2D())

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

