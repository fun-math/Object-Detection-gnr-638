# Object-Detection-gnr-638

This (incomplete) project aims to design an an architecture for perfoeming object detection in the context of autonomous driving. We combine ideas from the following 3 papers :
* [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934)
* [Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving](https://arxiv.org/pdf/1904.04620)
* [Object detection with location-aware deformable convolution and backward attention filtering](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Object_Detection_With_Location-Aware_Deformable_Convolution_and_Backward_Attention_Filtering_CVPR_2019_paper.pdf)

A major fraction of the implementation is taken from this repository https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch .

## Code Description 

### data

This folder contains train.txt and test.txt required for training for both COCO and KITTI datasets.

### src

This folder contains all the code for different parts of the model and for the dataset class.

### util

This folder contains a file for postprocessing of yolo output.

## How to run?

@Kartikey 

## References 

### Publication 

* [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934)
* [Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving](https://arxiv.org/pdf/1904.04620)
* [Object detection with location-aware deformable convolution and backward attention filtering](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Object_Detection_With_Location-Aware_Deformable_Convolution_and_Backward_Attention_Filtering_CVPR_2019_paper.pdf)

### Dataset 

* [COCO 2017](https://cocodataset.org/#download)
* [The KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)

### Implementation

* https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch
* https://github.com/Tianxiaomo/pytorch-YOLOv4