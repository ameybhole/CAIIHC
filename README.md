# Comparative analysis of individual identification of Holstein cattle

This respository contains code for the project "Comparative analysis of individual identification of Holstein cattle"

#### Summary

* [Introduction](#Introduction)
* [Getting Started](#Getting-started)
* [Dataset](#Dataset)
* [Repository Structure](#Repository-Structure)
* [Running Experiments](#Running-Experiments)
* [Performance](#Performance)
* [Conclusion](#Conclusion)

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

Link 1: http://www.cs.rug.nl/~george/cattle-recognition/

Link 2: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/O1ZBSA

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

## Running Expriments


## Performance

## Conclusion
