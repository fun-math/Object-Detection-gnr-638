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
- ladc.py is a module for location aware deformable convolutions.
- gaussianyololayer.py is a module which contains gaussian loss modelling.

### util

This folder contains a file for postprocessing of yolo output.

## How to run?

The folder src contains a train.py file which has an object hparams which contains training related information. This information needs to be set in the file (although there are default values). 
- "train_ds" will correspond to the location of train_images.txt file which contains location of training images
- "valid_ds" will correspond to the location of val_images.txt file which contains location of validation images

The training and validation images need to be placed in folders train_images and val_images respectively (or as mentioned in val_images.txt file) in the current working directory. Similarly, the training and validation labels should be placed in folders train_labels and val_labels respectively (or as mentioned in val_images.txt file) in the current working directory.

Finally, executing the file Object-Detection-gnr-638/src/train.py with the latest version of python will execute training.

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
