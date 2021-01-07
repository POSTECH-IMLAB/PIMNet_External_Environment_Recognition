# Traffic Light Recognition (TLR)
This TLR system designed as a module of Advanced Driver Assistance Systems (ADAS).
It detects and recognizes traffic lights from driving-view image sequences.

## Hyundai Contest (2017)
TLR system designed for Hyundai Contest (2017).
TLs are detected by using ACF and HUV-Histogram.

## Accurate Traffic Light Detection using Deep Neural Network with Focal Regression Loss (2018)
Based on RGB camera, implmented with Caffe.
For training from scratch::
train -solver tlr_yolov2_deconv_resnet101_focalloss4_4.prototxt

For training from weights::
train -solver tlr_yolov2_deconv_resnet101_focalloss4_4.prototxt -weights deconv_focal_tlr_iter_62000(best).caffemodel -gpu 0