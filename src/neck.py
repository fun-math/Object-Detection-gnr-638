import torch
from torch import nn
import torch.nn.functional as F

from convblock import *

class PAN_Layer(nn.Module):
    def __init__(self, in_channels, dropblock=True):#, sam=False, eca=False, ws=False, coord=False, hard_mish=False):
        super().__init__()

        in_c = in_channels
        out_c = in_c // 2

        self.c1 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        self.u2 = nn.Upsample(scale_factor=2, mode="nearest")
        # Gets input from d4
        self.c2_from_upsampled = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        # We use stack in PAN, so 512
        self.c3 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        self.c4 = ConvBlock(out_c, in_c, 3, 1, "leaky", dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.c5 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        self.c6 = ConvBlock(out_c, in_c, 3, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.c7 = ConvBlock(in_c, out_c, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)

    def forward(self, x_to_upsample, x_upsampled):
        x1 = self.c1(x_to_upsample)
        x2_1 = self.u2(x1)
        x2_2 = self.c2_from_upsampled(x_upsampled)
        # First is not upsampled!
        x2 = torch.cat([x2_2, x2_1], dim=1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.c6(x5)
        x7 = self.c7(x6)
        return x7

#Taken and modified from https://github.com/ruinmessi/ASFF/blob/0ff0e3393675583f7da65a7b443ea467e1eaed65/models/network_blocks.py#L267-L330
class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = ConvBlock(256, self.inter_dim, 3, 2, "leaky")
            self.stride_level_2 = ConvBlock(128, self.inter_dim, 3, 2, "leaky")
            self.expand = ConvBlock(self.inter_dim, 1024, 3, 1, "leaky")
        elif level==1:
            self.compress_level_0 = ConvBlock(512, self.inter_dim, 1, 1, "leaky")
            self.stride_level_2 = ConvBlock(128, self.inter_dim, 3, 2, "leaky")
            self.expand = ConvBlock(self.inter_dim, 512, 3, 1, "leaky")
        elif level==2:
            self.compress_level_0 = ConvBlock(512, self.inter_dim, 1, 1, "leaky")
            self.compress_level_1 = ConvBlock(256, self.inter_dim, 1, 1, "leaky")
            self.expand = ConvBlock(self.inter_dim, 256, 3, 1, "leaky")

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = ConvBlock(self.inter_dim, compress_c, 1, 1, "leaky")
        self.weight_level_1 = ConvBlock(self.inter_dim, compress_c, 1, 1, "leaky")
        self.weight_level_2 = ConvBlock(self.inter_dim, compress_c, 1, 1, "leaky")

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0 # 512 -> 512
            level_1_resized = self.stride_level_1(x_level_1) # 256 -> 512

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter) # 128 -> 512

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0) # 512 -> 256
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1 # 256 -> 256
            level_2_resized =self.stride_level_2(x_level_2) # 128 -> 256
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0) # 512 -> 128
            level_1_compressed = self.compress_level_1(x_level_1) # 256 -> 128
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2 #128 -> 128

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class Neck(nn.Module):
    def __init__(self, spp_kernels=(5, 9, 13), PAN_layers=[512, 256], dropblock=True,asff=False):#, sam=False, eca=False, ws=False, coord=False, hard_mish=False, asff=False):
        super().__init__()
        self.asff = asff

        self.c1 = ConvBlock(1024, 512, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        self.c2 = ConvBlock(512, 1024, 3, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.c3 = ConvBlock(1024, 512, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)

        # SPP block
        self.mp4_1 = nn.MaxPool2d(kernel_size=spp_kernels[0], stride=1, padding=spp_kernels[0] // 2)
        self.mp4_2 = nn.MaxPool2d(kernel_size=spp_kernels[1], stride=1, padding=spp_kernels[1] // 2)
        self.mp4_3 = nn.MaxPool2d(kernel_size=spp_kernels[2], stride=1, padding=spp_kernels[2] // 2)

        self.c5 = ConvBlock(2048, 512, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        self.c6 = ConvBlock(512, 1024, 3, 1, "leaky", dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.c7 = ConvBlock(1024, 512, 1, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)

        self.PAN8 = PAN_Layer(PAN_layers[0], dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.PAN9 = PAN_Layer(PAN_layers[1], dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)

        if asff: # branch inputs biggest objects: 512, medium objects: 256, smallest objects : 128
            self.ASFF_0 = ASFF(0)
            self.ASFF_1 = ASFF(1)
            self.ASFF_2 = ASFF(2)

    def forward(self, input):
        d5, d4, d3 = input

        x1 = self.c1(d5)
        x2 = self.c2(x1)
        x3 = self.c3(x2)

        x4_1 = self.mp4_1(x3)
        x4_2 = self.mp4_2(x3)
        x4_3 = self.mp4_3(x3)
        x4 = torch.cat([x4_1, x4_2, x4_3, x3], dim=1)

        x5 = self.c5(x4)
        x6 = self.c6(x5)
        x7 = self.c7(x6)

        x8 = self.PAN8(x7, d4)
        x9 = self.PAN9(x8, d3)

        if self.asff:
            x7 = self.ASFF_0(x7)
            x8 = self.ASFF_1(x8)
            x9 = self.ASFF_2(x9)

        return (x9, x8, x7)