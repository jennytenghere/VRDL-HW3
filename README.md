# VRDL-HW3
StudentID: 313553027  
Name: 鄧婕妮
## Introduction
This is the third assignment of the VRDL course. The task is to apply Mask RCNN (or other model) to the pathology dataset, tackling classification, detection, and segmentation tasks simultaneously. I tried to modify some internal parameters of the original Mask R-CNN model, including replacing the default ResNet backbone with ResNeXt, and experimenting with deeper mask predictor heads. For details on the differences between each version, please refer to the report. Among all versions, v1 achieved the highest validation mAP.

## How to install and use
### Install environment
conda env create -f vrdl.yaml
### run training and validation
python 250422_v1.py
### run test (output submission file)
python 250428_test_v1.py

## Performance
![image](https://github.com/user-attachments/assets/1ce1fd80-8c50-4ce1-9c32-6e3865270e90)
