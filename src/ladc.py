from __future__ import absolute_import, division

import torch
import torch.nn as nn

import numpy as np
from deformable import th_batch_map_offsets, th_generate_grid
from convblock import *


class ConvOffset2D(nn.Conv2d):
    """ConvOffset2D
    Convolutional layer responsible for learning the 2D offsets and output the
    deformed feature map using bilinear interpolation
    Note that this layer does not perform convolution on the deformed feature
    map. See get_deform_cnn in cnn.py for usage
    """
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init
        Parameters
        ----------
        filters : int
            Number of channel of the input feature map
        init_normal_stddev : float
            Normal kernel initialization
        **kwargs:
            Pass to superclass. See Con2d layer in pytorch
        """
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))
        self.pad = nn.ZeroPad2d(1)

    def forward(self, x):
        """Return the deformed featured map"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)

        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)

        # X_offset: (b*c, h, w)
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))

        # x_offset: (b, h, w, c)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        x_offset = self.pad(x_offset)

        return x_offset

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x



class DeformConvNet(nn.Module):
    def __init__(self, ch,  dropblock=True):
        super().__init__()
        self.c0 = ConvBlock(ch, 64, 1, 1, 'mish', dropblock=dropblock)
        self.c1 = ConvOffset2D(64)
        self.c2 = ConvOffset2D(64)
        self.c3 = ConvOffset2D(64)
        self.c4 = ConvOffset2D(64)
        self.c5 = ConvOffset2D(64)
        self.c6 = ConvOffset2D(64)
        self.c7 = ConvOffset2D(64)
        self.c8 = ConvOffset2D(64)
        self.c9 = ConvOffset2D(64)

        self.c3d = nn.Conv3d(64,64,(1,1,9))
        self.cl = ConvBlock(ch+64, ch, 1, 1, 'mish', dropblock=dropblock)

        if dropblock:
            self.use_dropblock = True
            self.dropblock = DropBlock2D()
        else:
            self.use_dropblock = False


    def forward(self, x):
        h = x
        x = self.c0(x)
        shape = x.size()
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x5 = self.c5(x)
        x6 = self.c6(x)
        x7 = self.c7(x)
        x8 = self.c8(x)
        x9 = self.c9(x)

        y = torch.stack([x1[:,:,0:,0:],x1[:,:,0:,1:],x1[:,:,0:,2:],x1[:,:,1:,0:],x1[:,:,1:,1:],x1[:,:,1:,2:],x1[:,:,2:,0:],x1[:,:,2:,1:],x1[:,:,2:,2:]],dim=4)
        y = self.c3d(y).view(shape)

        x = torch.cat([h,y],dim=1)
        x = cl(x)

        if self.use_dropblock:
            x = self.dropblock(x)

        return x