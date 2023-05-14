# DogEmotionClassifier

## Overview
This repository contains code for a deep learning model that can classify the emotions of dogs in images. The model is based on the MobileNet architecture, and is designed to be lightweight and easy to host as an API.

## MobileNet Architecture
MobileNet is a popular convolutional neural network (CNN) architecture that was specifically designed for mobile and embedded devices. It uses depthwise separable convolutions to reduce the number of parameters and computation required for inference, while still maintaining a high level of accuracy. This makes it an ideal choice for our dog emotion classifier, as it allows us to create a model that is both accurate and easy to deploy.

## Optimizers and Data Augmentation
To prevent overfitting, we used the Adam optimizer, which is known for its robustness and efficiency in training deep neural networks. We also used data augmentation techniques such as random rotations, flips, and zooms, to increase the size of our training dataset and reduce overfitting.

## Training Loss Graphs
The following graphs show the training and validation loss curves for the model:

# Training Loss v/s Validation Loss Graph

# Training Accuracy V/s Validation Accuracy Graph

## Data Set
The original data set used to train this model can be found at the following link: https://www.kaggle.com/datasets/danielshanbalico/dog-emotion

## Proof of Work
A video demonstrating the training and evaluation of the model can be found at the following link: https://youtu.be/JnMA1M1s8tA

## Usage
To use this model as an API, simply install the required dependencies and run the .py file.

# Dependencies
TensorFlow
Flask
Python 3.9 or above
Sklearn

Credits
This project was created by [Dhruv Sharma].
