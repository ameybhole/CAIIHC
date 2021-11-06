# Comparative analysis of individual identification of Holstein cattle

This respository contains code for the project "Comparative analysis of individual identification of Holstein cattle"

#### Summary

* [Introduction](#Introduction)
* [Getting Started](#Getting-started)
* [Dataset](#Dataset)
* [Repository Structure](#Repository-Structure)
* [Running Experiments](#Running-Experiments)
* [Performance](#Performance)
<!-- * [Conclusion](#Conclusion) -->

## Introduction

The monitoring of cattle in farms is important as it allows farmers keeping track of the performance indicators and any signs of health issues. In Europe, bovine identification is mostly dependent upon the electronic ID/RFID ear tags, as opposed to branding and tattooing. The ear-tagging approach has been called into question because of physical damage and animal welfare concerns. In this project, we perform a comparative analysis on individual identification of cattle using transfer learning based on models trained on ImageNet and ms-celeb1million. We perform transfer learning using the pre-trained models ResNet50, FaceNet, VGG-16 and VGG-Face for individual identification of Holstein cattle. We apply data augmentation techniques such as Rotation (+/-10 degrees), Flip, Gaussian noise and illumination to test its effects on the classification and use T-SNE for visualization. We conduct experiments using 5-fold cross classification on a dataset consisting of 1237 segmented RGB images with 136 unique classes (9 images per class)

## Getting started

In order to run this repository, we advise you to install python 3.6 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install requirments on it:

```bash
conda create --name caiihc python=3.6
```

Clone this repository:

```bash
git clone https://github.com/ameybhole/CAIIHC.git 
```

Install requirements:

```bash
$ pip install -r requirements.txt
```

## Dataset

The data set consists of 1237 pairs of thermal and RGB (640 x 320 pixels and 320 x 240 pixels) images with 136 classes (i.e. 136 different cows) with a mean of 9 images per class/cow. Each folder name is the collar id of the cattle and contains its respective thermal and RGB images. The data set was collected at the Dairy Campus in Leeuwarden, The Netherlands. In order to explore the temperature values of the thermal images, FLIR tools can be used by installing the software (https://www.flir.com/products/flir-tools/).

The dataset can be found here: 

- Link 1: http://www.cs.rug.nl/~george/cattle-recognition/

- Link 2: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/O1ZBSA

The pre-processing steps used were based on the following paper:

[A. Bhole, O. Falzon, M. Biehl, G. Azzopardi, “A Computer Vision Pipeline that Uses Thermal and RGB Images for the Recognition of Holstein Cattle”, Computer Analysis of Images and Patterns (CAIP), pp. 108-119, 2019](https://link.springer.com/chapter/10.1007/978-3-030-29891-3_10)

Github link to the code for the above paper: 

- Link: https://github.com/ameybhole/IIHC

## Repository Structure

```
.
├── src                        # source code of the project 
|   ├── data_augmentation.py   # script for data augmentation
|   ├── data_load.py           # script for data loading
|   ├── evaluation.py          # script for model evaluation
|   ├── models.py              # script for compiling model
|   ├── train.py               # script to train model
|   └──visualizations.py       # script for visualization
├── weights                    # weights of FaceNet model 
|   ├── facenet_keras.h5             # model
|   └── facenet_keras_weights.h5     # weights of model
└── main.py                    # script to run exepriments with different parameters settings
```

FaceNet weights and model were downloaded from the following github repository:

https://github.com/nyoki-mtl/keras-facenet

## Running Experiments

#### Train without augmentation

```Bash
python main.py --num_epochs 20 --mode train --dataset [path to dataset] --resize 224 --tsne False --batch_size 32 --classes 136 --trainable True --include_top False --model resent50 
```
#### Train with augmentation

```Bash
python main.py --num_epochs 20 --mode augment --dataset [path to dataset] --resize 224 --tsne False --batch_size 32 --classes 136 --trainable True --include_top False --flip True --rotation_left -10 --rotation_right 10 --bright 1.5 --dark 0.5 --gaussian_nosie True --model resent50
```

## Performance

#### Results without augmentation

|          Model         |   Accuracy   | Precision  | Recall |  
| ---------------------- | ------------ | ------------ | ------------ | 
| `ResNet50`              | `98.31 ± 0.067` | `98.78 ± 0.023` | `98.21 ± 0.072` | 
| `FaceNet`              | `98.79 ± 0.097` | `99.01 ± 0.061` | `98.78 ± 0.083` | 
| `VGG-16`              | `90.11 ± 0.085` | `91.12 ± 0.025` | `91.45 ± 0.022` | 
| `VGG-Face`              | `93.25 ± 0.077` | `93.44 ± 0.084` | `93.22 ± 0.075` | 

#### Results with augmentation

|          Model         |   Accuracy   | Precision  | Recall |  
| ---------------------- | ------------ | ------------ | ------------ | 
| `ResNet50`              | `98.23 ± 0.035` | `98.10 ± 0.060` | `97.42 ± 0.095` | 
| `FaceNet`              | `98.55 ± 0.0097` | `98.33 ± 0.035` | `97.98 ± 0.081` | 
| `VGG-16`              | `91.22 ± 0.023` | `91.75 ± 0.085` | `91.36 ± 0.064` | 
| `VGG-Face`              | `93.66 ± 0.034` | `93.45 ± 0.074` | `93.64 ± 0.071` | 

