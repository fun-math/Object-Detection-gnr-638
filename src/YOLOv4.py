import torch
from torch import nn
import torch.nn.functional as F

from convblock import *
from backbone import *
from neck import *
from head import *
from yololayer import *
from gaussianyololayer import *

class YOLOv4(nn.Module):
    def __init__(self, in_channels=3, n_classes=80, weights_path=None, pretrained=False, img_dim=608, anchors=None, dropblock=True, iou_aware=None, deformable=deformable, gaussian_loss=None):#, sam=False, eca=False, ws=False, iou_aware=False, coord=False, hard_mish=False, asff=False, repulsion_loss=False):
        super().__init__()
        if anchors is None:
            anchors = [[[10, 13], [16, 30], [33, 23]],
                       [[30, 61], [62, 45], [59, 119]],
                       [[116, 90], [156, 198], [373, 326]]]

        output_ch = (4 + 1 + n_classes) * 3
        #if iou_aware:
            #output_ch += 1 #1 for iou

        if gaussian_loss is not None :
            output_ch+=3

        self.img_dim = img_dim

        self.backbone = Backbone(in_channels, dropblock=False, deformable=deformable)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)

        self.neck = Neck(dropblock=dropblock)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish, asff=asff)

        self.head = Head(output_ch, dropblock=False)#, sam=sam, eca=eca, ws=ws, coord=coord, hard_mish=hard_mish)

        if gaussian_loss is not None :
            self.yolo1 = GaussianYOLOLayer(anchors[0], n_classes, img_dim, iou_aware=iou_aware)#, type=gaussian_loss)#, repulsion_loss=repulsion_loss)
            self.yolo2 = GaussianYOLOLayer(anchors[1], n_classes, img_dim, iou_aware=iou_aware)#, type=gaussian_loss)#, repulsion_loss=repulsion_loss)
            self.yolo3 = GaussianYOLOLayer(anchors[2], n_classes, img_dim, iou_aware=iou_aware)#, type=gaussian_loss)#, repulsion_loss=repulsion_loss)


        else :
            self.yolo1 = YOLOLayer(anchors[0], n_classes, img_dim, iou_aware=iou_aware)#, repulsion_loss=repulsion_loss)
            self.yolo2 = YOLOLayer(anchors[1], n_classes, img_dim, iou_aware=iou_aware)#, repulsion_loss=repulsion_loss)
            self.yolo3 = YOLOLayer(anchors[2], n_classes, img_dim, iou_aware=iou_aware)#, repulsion_loss=repulsion_loss)

        if weights_path:
            try:  # If we change input or output layers amount, we will have an option to use pretrained weights
                self.load_state_dict(torch.load(weights_path), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')
        elif pretrained:
            try:  # If we change input or output layers amount, we will have an option to use pretrained weights
                self.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch/releases/download/V1.0/yolov4.pth"), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')

    def forward(self, x, y=None):
        b = self.backbone(x)
        n = self.neck(b)
        h = self.head(n)

        h1, h2, h3 = h

        out1, loss1 = self.yolo1(h1, y)
        out2, loss2 = self.yolo2(h2, y)
        out3, loss3 = self.yolo3(h3, y)

        out1 = out1.detach()
        out2 = out2.detach()
        out3 = out3.detach()

        out = torch.cat((out1, out2, out3), dim=1)

        loss = (loss1 + loss2 + loss3)/3

        return out, loss
