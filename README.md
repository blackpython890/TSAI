# Applied Deep Learning : Convolution Neural Network 
<p align="center">

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)&nbsp;&nbsp;&nbsp;[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)&nbsp;&nbsp;&nbsp;[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)&nbsp;&nbsp;&nbsp;[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

</p>

This repo contains source codes for, Building Intuition for Convolution Neural Networks step by step.

<details>
    <summary>Session-01</summary>

- Machine Learning Intuition, Background & Basics
- Python 101 for Machine Learning
- [blog](https://myselfhimanshu.github.io/posts/cnn_01/)

</details>

<details>
    <summary>Session-02</summary>

- Convolutions, Pooling Operations & Channels
- Pytorch 101 for Vision Machine Learning
- [blog](https://myselfhimanshu.github.io/posts/cnn_02/)

</details>

<details>
    <summary>Session-03</summary>

- Kernels, Activations and Layers
- [blog](https://myselfhimanshu.github.io/posts/cnn_03/)

</details>

<details>
    <summary>Session-04</summary>

- Architectural Basics suitable for our objective
- MNIST model training 
    - parameters used 13,402
    - epochs=20
    - highest test accuracy = 99.46%, epoch = 19th
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-04/final_submission/MNIST_model_04.ipynb)

</details>

<details>
    <summary>Session-05</summary>

- Receptive Field : core fundamental concept
- MNIST model training
    - parameters used 7808
    - epochs=15
    - highest test accuracy = 99.43%, epoch = 11th 
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-05/after-work/MNIST_model_final.ipynb)

</details>

<details>
    <summary>Session-06</summary>

- BN, Kernels & Regularization
- MNIST model training
    - using L1/L2 regularization with BN/GBN
    - BN : batch normalization
    - GBN : ghost batch normalization
    - best model : BN with L2
        - parameters used 7808
        - epochs=25
        - highest test accuaracy = 99.54%, epoch = 21st
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-06/notebooks/MNIST_model_regularization.ipynb)

</details>

<details>
    <summary>Session-07</summary>

- Advanced Convolution
- Achieve an accuracy of greater than 80% on CIFAR-10 dataset
    - architecture to C1C2C3C40 (basically 3 MPs)
    - total params to be less than 1M
    - RF must be more than 44
    - one of the layers must use Depthwise Separable Convolution
    - one of the layers must use Dilated Convolution
    - use GAP
- Result
    - parameters used 220,778
    - epochs = 20
    - highest test acc = 85.55%
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-07/notebooks/002_main_85_55.ipynb)

</details>

<details>
    <summary>Session-08</summary>

- Receptive Fields and Network Architectures
- Achieve an accuracy of greater than 85% on CIFAR-10 dataset
    - architecture ResNet18
- Result
    - parameters : 11,173,962
    - epoch : 50
    - training acc : 98.65%
    - testing acc : 89.78%
    - [notebook link](https://github.com/myselfHimanshu/ai-vision-program/blob/master/Session-08/notebooks/001_main_89_78.ipynb)

</details>

<details>
    <summary>Session-09</summary>

- Data Augmentation
- Achieve an accuracy of greater than 87% on CIFAR-10 dataset
    - Move transformations to Albumentations. 
    - Implement GradCam function. 
- Result
    - parameters : 11,173,962
    - epoch : 50
    - testing acc : 92.17%
    - [work link](https://github.com/myselfHimanshu/ai-vision-program/tree/master/Session-09)

</details>

<details>
    <summary>Session-10</summary>

- Advanced Concepts : Training and Learning Rates
- Achieve an accuracy of greater than 88% on CIFAR-10 dataset
    - Add CutOut augmentation
    - Implement LR Finder (for SGD, not for ADAM)
    - Implement ReduceLROnPlateau
- Result
    - parameters : 11,173,962
    - epoch : 50
    - testing acc : 89.80%
    - [work link](https://github.com/myselfHimanshu/ai-vision-program/tree/master/Session-10)

</details>

<details>
    <summary>Session-11</summary>

- Super Convergence
- Achieve an accuracy of greater than 90% on CIFAR-10 dataset
    - 3Layer-Network
    - Implement One Cycle Policy
- Result
    - parameters : 6,573,130
    - epoch : 24
    - testing acc : 91.02%
    - [work link](https://github.com/myselfHimanshu/ai-vision-program/tree/master/Session-11)

</details>

<details>
    <summary>Session-12</summary>

- Object Localization : YOLO (Recognition)
- Use TinyImageNet dataset, create custom data loader with 70/30 split.
- Achieve an accuracy of greater than 50% on TinyImageNet dataset
    - ResNet18
    - One Cycle Policy
- Result
    - parameters : 11,173,962
    - epoch : 30
    - testing acc : 58.35%
    - [work link](https://github.com/myselfHimanshu/ai-vision-program/tree/master/Session-12)

</details>

