import torch
from torch import nn
import torch.nn.functional as F

from convblock import *

###############################################
#Takes a tuple as input from neck module
#Returns a tuple as output
###############################################

class HeadPreprocessing(nn.Module):
    def __init__(self, in_channels, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False):
        super().__init__()
        ic = in_channels
        self.c1 = ConvBlock(ic, ic*2, 3, 2, 'leaky', dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.c2 = ConvBlock(ic*4, ic*2, 1, 1, 'leaky', dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        self.c3 = ConvBlock(ic*2, ic*4, 3, 1, 'leaky', dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.c4 = ConvBlock(ic*4, ic*2, 1, 1, 'leaky', dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)
        self.c5 = ConvBlock(ic*2, ic*4, 3, 1, 'leaky', dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.c6 = ConvBlock(ic*4, ic*2, 1, 1, 'leaky', dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=False, hard_mish=hard_mish)

    def forward(self, input, input_prev):
        x1 = self.c1(input_prev)
        x1 = torch.cat([x1, input], dim=1)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        x6 = self.c6(x5)

        return x6


class HeadOutput(nn.Module):
    def __init__(self, in_channels, out_channels, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False):
        super().__init__()
        self.c1 = ConvBlock(in_channels, in_channels*2, 3, 1, "leaky", dropblock=False)#, sam=sam, eca=eca, ws=False, coord=coord, hard_mish=hard_mish)
        self.c2 = ConvBlock(in_channels*2, out_channels, 1, 1, "linear", bn=False, bias=True, dropblock=False)#, sam=False, eca=False, ws=False, coord=False, hard_mish=hard_mish)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        return x2


class Head(nn.Module):
    def __init__(self, output_ch, dropblock=True, sam=False, eca=False, ws=False, coord=False, hard_mish=False):
        super().__init__()

        self.ho1 = HeadOutput(128, output_ch, dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)

        self.hp2 = HeadPreprocessing(128, dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.ho2 = HeadOutput(256, output_ch, dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)

        self.hp3 = HeadPreprocessing(256, dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)
        self.ho3 = HeadOutput(512, output_ch, dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)

    def forward(self, input):
        input1, input2, input3 = input

        x1 = self.ho1(input1)
        x2 = self.hp2(input2, input1)
        x3 = self.ho2(x2)

        x4 = self.hp3(input3, x2)
        x5 = self.ho3(x4)

        return (x1, x3, x5)
'''
class Head(nn.Module):
	def __init__(self,out_ch,n_classes,inference=False):
		super().__init__()
		self.inference=inference

		self.conv1=Conv_Bn_Act(128,256,3,1,'leaky')
		self.conv2=Conv_Bn_Act(256,out_ch,1,1,'linear',bn=False)



		self.conv3=Conv_Bn_Act(128,256,3,2,'leaky')

		self.conv4 = Conv_Bn_Act(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Act(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Act(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Act(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Act(512, 256, 1, 1, 'leaky')

        self.conv9 = Conv_Bn_Act(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Act(512, output_ch, 1, 1, 'linear', bn=False)



        self.conv11 = Conv_Bn_Act(256,512,3,2,'leaky')

        self.conv12 = Conv_Bn_Act(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Act(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Act(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Act(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Act(1024, 512, 1, 1, 'leaky')
        
        self.conv17 = Conv_Bn_Act(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Act(1024, output_ch, 1, 1, 'linear', bn=False)

    def forward(self,in1,in2,in3):
        x1 = self.conv1(in1)
        x2 = self.conv2(x1)


        x3 = self.conv3(in1)
        
        x3 = torch.cat([x3, in2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        
        x11 = self.conv11(x8)
        
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            return [x2, x10, x18]
'''