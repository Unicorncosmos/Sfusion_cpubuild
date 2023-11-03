# SFusion CPU Version Readme

segmentation_model: Internimage
Ops: DCNv3
Transformer: LSSviewTransformer

This repository will generate result.pkl fusion data

## Overview
The InternImage backbone with dcnv3 has cpu build capabilities 
also bevpool and lssview transformer has cpu build capabilities 
This readme provides instructions for setting up and using the CPU version of the SFusion model with DCNV3, LSSViewTransformer, and DepthNet on the NuScenes dataset. SFusion is a sensor fusion model designed to work with radar and camera data for 3D object detection and depth estimation.

## Table of Contents

- [Prerequisites](#prerequisites)

## Prerequisites

```shell
mmcv=1.4.0=pypi_0
mmcv-full=1.4.0=pypi_0
mmdet=2.28.1=dev_0
mmengine=0.8.2=pypi_0
mmsegmentation=0.30.0=pypi_0  
onnx=1.14.1=pypi_0
onnxruntime=1.16.0=pypi_0
pytorch=1.9.0=py3.9_cpu_0
```

## To run this model on CPU

To run this model on CPU, you will need
Either 

1. A machine with a CPU that supports AVX2 instructions
2. A machine with a CPU that has minimum 16GB RAM.
```
Install the Nuscenes Dataset
```

Use the [mmdet3d](https://github.com/HVXLab/HVDetFusion/tree/main/mmdet3d) HVDetFusion repo for dataloader and dataprocessing pipeline


```
Install the onnx model
```
```
pip install -r requiremets.txt
```
or
```
docker build -t sfusion_cpu .
```

