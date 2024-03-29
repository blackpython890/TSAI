[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)


# Applied Deep Learning : Convolution Neural Network

[![Python](https://img.shields.io/badge/Language%20%26%20Version-Python%203.6%2B-brightgreen?logo=python)](https://www.python.org/)&nbsp;&nbsp;[![PyTorch](https://img.shields.io/badge/Library-PyTorch-brightgreen?logo=pytorch)](https://pytorch.org)&nbsp;&nbsp;![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen?logo=github)&nbsp;&nbsp;[![License](https://img.shields.io/badge/LICENSE-MIT-brightgreen)](https://github.com/jagatabhay/TSAI/blob/master/LICENSE)&nbsp;&nbsp;![LAST_COMMIT](https://img.shields.io/github/last-commit/jagatabhay/TSAI)&nbsp;&nbsp;![Contributors](https://img.shields.io/github/contributors/jagatabhay/TSAI?style=plastic)&nbsp;&nbsp;


## What this REPO is about


With the advancement in the field of GPU and deep learning, the task of the model to classify the images,object detection etc is now something we can acheive with relatively higher accuracy and in real-time. And with this branch of Artifical Intelligence ***Computer Vision*** started to evolve. One of the driving factors behind the growth of Computer Vision field is the amount of data(images , videos ) we generate today is sufficient to trian the Convolution Neural Network (CNN) and make Computer Vision better.

This repository contains my personal exploration and research on Convolution Neural Network , disciplined way to learn and implement the fundamentals of State Of the Art models using PyTorch library.

<p align = 'center'>
  <img width = '600' height = '350' src = "https://github.com/jagatabhay/miscellaneous/blob/master/humanpose.gif">
 </p>


---

<p align="center">
  <b> Lets Get Started </b><br>
  <img width="400" height="300" src="https://github.com/jagatabhay/miscellaneous/blob/master/gettingstartedlogo.png">
</p>



### Prerequisite
- Python
- Knowledge of Image( heights , width , pixels , channels)
- PyTorch
- Basics of OpenCV , Python Image Library(PIL) , matplotlib.


### Hardware Requirement
- GPU : Tesla T4/Tesla K8 or higher versions.
- GPU count : 1,2
- RAM - 12GB or higher


### Models Used
- [x] Custom Deep Neural Network
- [x] ResNet
- [x] DenseNet
- [ ] GoogleNet

### Case Study
- [x] Image Classification.
- [x] Object Detection using YOLO.
- [x] Monocular Depth Estimation.
- [ ] Object Segmentation.
- [ ] Human Pose Estimation.
- [ ] GAN's

### Dataset Used
- [x] MNIST
- [x] CIFAR10
- [x] Custom Dataset for Object Detection.
- [x] Tiny ImageNet
- [ ] Coco
- [ ] ImageNet

---

<details>
  <summary>1. ML Intuition and Basics of CNN </summary>
   
Basics of python can be learnt on YouTube. Channels like Corey Shagffer [![YouTubeLogo](https://github.com/jagatabhay/TSAI/blob/master/S13/logo.png)](https://www.youtube.com/playlist?list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU) and Telusko [![YouTubeLogo](https://github.com/jagatabhay/TSAI/blob/master/S13/logo.png)](https://www.youtube.com/c/Telusko/playlists) helped me a lot to learn about python basics.


Basics of CNN , how CNN learns , how different channels are formed , how DNN make sense of the inputs it gets ( __Features -> Edges & Gradients -> Textures -> Patterns -> Part of Objects -> Objects -> Scenes__ )Please see below. Resemblance of Human brain , eyes with computer vision field.

<p align='center'>
  <img src="https://github.com/jagatabhay/miscellaneous/blob/master/Edges%20and%20Gradient.PNG">
</p>

</details>

<details>
  <summary>2. CNN Architecture </summary>
   
Basic CNN Architecture , maintain symmetry by chosing odd size kernel(Example : 3X3 , 5X5), importance of choosing 3X3 kernel over 5X5 or higher odd kernel , Max-Pooling  , Receptive Field. Below image represents convolution from 5x5-3x3-1x1 and receptive field increase from left to right as convolution occurs or layers increases.
  
<p align='center'>
  <img src= 'https://github.com/jagatabhay/miscellaneous/blob/master/RF.gif'>
</p>
 
</details>
 
<details>
  <summary>3. Kernels and Convolution </summary>
  
Basic Pytorch architecture for working with neural networks, introduction to nn.Module, optimizers, forward and backward pass, datasets, how to apply simple augmentation.
 </details>


<details>
<summary>4. Architecture Basics</summary>

CNN Architecture components Fully Connected Layer , Drop-Out , Softmax , Learning-Rate , Batch-Size.

Work link Summary :
- Train MNIST Dataset to get 99.40% accuracy with given contraint. Kindly check the [worklink](https://github.com/jagatabhay/S4) to know more.
- Parameters : 
- Epoch : 20
- Learning Rate
- Batch Size 
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/S4)

Fully Connected layer(FC) vs Drop-Out vs Learning Rate is shown below respectively.
<p align='center'>
  <img width = 300 height = 200 src= 'https://github.com/jagatabhay/miscellaneous/blob/master/fullyconnectedlayer.png'>
  <img width = 300 height = 200 src= 'https://github.com/jagatabhay/miscellaneous/blob/master/droput.gif'>
  <img width = 300 height = 200 src= 'https://github.com/jagatabhay/miscellaneous/blob/master/LR.jpg'>
  </p>

</details>



<details>
    <summary>5. Model Implementation</summary>
    
Step by step approach to build neural network , debugg , and to optimize to get the best accuracy.
Kindly check [worklink](https://github.com/jagatabhay/S5) to know more.

Work link Summary :
- Train MNIST Dataset to get 99.40% accuracy with given contraint. Kindly check the [worklink](https://github.com/jagatabhay/S5) to know more.
- Parameters : 
- Epoch : 15
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/S5)
</details>



<details>
<summary>6. Batch Normalization and Regularization</summary>
  

Importance of Normalization , Batch normalization , Regularization of Datasets. Thin line difference between normalization and equalization.
 
Work link Summary :
- Train MNIST Dataset to get 99.40% accuracy with contraint and add regularization to it.Kindly check the [worklink](https://github.com/jagatabhay/TSAI/tree/master/S6) to know more.
- Parameters : 
- Epoch : 15
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S6)

Original Data Mean vs Normalized Data mean is shown below recpectively.
<p align = 'center'>
  <img width = 400 height = 400 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/normalization.png'>
 </p>
 
 </details>


<details>
<summary>7. Advanced Convolutions </summary>

Different Types of convolution like Normal Convoultion, Dilated Convolutions, Pointwise Convolution(1x1), DECONVOLUTION or Fractionally Strided OR Transpose Convolution, Pixel Shuffle Algorithm, Depthwise Separable Convolution, Grouped Convolution. Dilated, Depthwise , Grouped is shown below respectively.

Work link Summary :
- Train CIFAR10 Dataset to get more that 80% accuracy with contraints.Kindly check the [worklink](https://github.com/jagatabhay/TSAI/tree/master/S7) to know more.
- Parameters : 
- Epoch : 15
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S7)

Dilated Convolution vs Depthwise vs Group Convolution is shown below respectively.
<p align = 'center'>
  <img width = 350 height = 250 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/dilatedConvulation.gif'>
  <img width = 350 height = 250 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/depthwise.png'>
  <img width = 350 height = 250 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/groupedconvulation.png'>
 </p> 
</details>


<details>
<summary>8. Receptive Fields and dfferent Netwwork Architecture </summary>

Introduction to different neural network architecture like AlexNet , VGG , ResNet, GoogleNet, Inception, ResNext. Different Version of it. Importance of having multiple Receptive field.

Work link Summary :
- Train CIFAR10 Dataset to get more that 85% accuracy using ResNet-18 architecture. Kindly check the [worklink](https://github.com/jagatabhay/TSAI/tree/master/S8) to know more.
- Model :
- Epoch : 15
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S8)

Comparison of architecture like AlexNet, VGGNet, ResNet is shown below.

<p align = 'center'>
  <img width = 400 height = 400 src='https://github.com/jagatabhay/miscellaneous/blob/master/AlexNet.png'>
  <img width = 400 height = 400 src='https://github.com/jagatabhay/miscellaneous/blob/master/VGGNet.png'>
  <img width = 400 height = 400 src='https://github.com/jagatabhay/miscellaneous/blob/master/ResNet.png'>
</p>

</details>


<details>
<summary>9. Data Augmentation/Model Diagnostics </summary>

One of the easy way to increase accuracy is to increase the receptive field(core idea of ResNet architecturec). One of the way also include regularization like DropOut , Batch Normalization , L1/L2 Regularization. All the above topic will fall short if the dataset is limited. And to tackle this we can use __Data Augmentation__ strategy.
Please see some the strategy mentiond images.

Work link Summary :
- Implement Augmentation module , GRADCAM module. And train the CIFAR10 dataset to achieve 87%+ accuracy. Kindly check the [worklink](https://github.com/jagatabhay/TSAI/tree/master/S9) to know more.
- Model :
- Epoch : 15
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link]https://github.com/jagatabhay/TSAI/tree/master/S9)

Just have a look at different data augmentation strategy.
<p align = 'center'>
  <img width = 300 , height = 300 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/DA1.png'>
  <img width = 300 , height = 300 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/DA2.png'>
  <img width = 300 , height = 300 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/DA3.png'>
</p>

</details>


<details>
<summary>10. Advanced Training </summary>

LR Finder.
This need to update.
[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S10)
</details>


<details>
<summary>11. Super-Convergence </summary>

Implementation of phenomenon( Super-Convergence/One Cycle Policy) where a neural network can be trained on a faster magnitude than a standard training without hampering accuracy of the model.This is the implementation of reasearch paper [discussed here](https://arxiv.org/pdf/1708.07120.pdf). An intuition to implement this is that large learning rates regularize the training, hence requiring a reduction of all other forms of regularization in order to preserve the optimal balance. 

Work link Summary :
- Implement one-cycle policy along with data-augmentation strategy ad show GRADCAM module. And train the CIFAR10 dataset to achieve 90%+ accuracy. Kindly check the [worklink](https://github.com/jagatabhay/TSAI/tree/master/S11) to know more.
- Model :
- Epoch : 15
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S11)

One Cycle Minima , Test accuracy to show significance of Super-convergnece.

<p align = 'center'>
  <img width = 250 , height = 250 src= 'https://github.com/jagatabhay/miscellaneous/blob/master/OneCycleMinima.png'>
  <img width = 500 , height = 300 src= 'https://github.com/jagatabhay/miscellaneous/blob/master/OneCyclePolicy.png'>
 </p>
 
</details>


<details>
<summary>12. Object Localization </summary>

Difference between Image classification and Image localization ( aka Image/Object Detection ). Detection approaches like Sliding window alogorithm, Regional propasal algorithms, Anchor box,shown below respectively. Pros and Cons of different approaches. Detailed study of latest approach anchor box - IOU ( Intersection over Union ), MAP ( Mean Aeverage Precision ), centriods, K-Means algorithms to compute centroids. Understanding YOLO-V2 [loss function](https://www.meetup.com/Machine-Learning-India-Bangalore/messages/boards/thread/52385226).

Work link Summary :
- Train Tiny-ImageNet on ResNet-18 within contraint to acheive 50%+ accuracy. [worklink](https://github.com/jagatabhay/TSAI/tree/master/S12) to know more.
- Model : ResNet-18
- Epoch : 50
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S12)

Lets have a visualization of Sliding Window algorithm , Regional Proposal, Anchor Box.

<p align = 'center'>
  <img width = 300 height = 300 src='https://github.com/jagatabhay/miscellaneous/blob/master/SlidingWindow.gif'>
  <img width = 300 height = 300 src='https://github.com/jagatabhay/miscellaneous/blob/master/RegionalProposal.gif'>
  <img width = 300 height = 300 src='https://github.com/jagatabhay/miscellaneous/blob/master/AnchorBox.gif'>
</p>

</details>



<details>
<summary>13. YOLO 2 & 3 </summary>

Introduction to YOLO and why is it called YOLO ? FPS of YOLO. Anchor Box variation on datasets.

Work link Summary :
- Use OpenCV to detect COCO Dataset Objects. Collect a custom dataset of 500 Images and detection by YOLO. [worklink](https://github.com/jagatabhay/TSAI/tree/master/S13) to know more.
- Model : ResNet-18
- Epoch : 50
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S13)
- Object Detection Youtube Video - [![YouTube](https://github.com/jagatabhay/TSAI/blob/master/S13/logo.png)](https://www.youtube.com/watch?v=A0n0CvoeFEI)

<p align = 'center'>
  <img width = 350 height = 350 src = 'https://github.com/jagatabhay/miscellaneous/blob/master/YOLO.png'>
 </p>
 
</details>


<details>
  <summary>14. RCNN </summary>
  
<p align = 'center'>
  <img width = 600 height = 300 src= 'https://github.com/jagatabhay/miscellaneous/blob/master/MaskRCNN.png'>
 </p>
 
Introduction to RCNN family. RCNN family find it's root in  [Selective Search for Object Recognition - SSOR](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf) and [Efficient Graph based Image Segmentation - EGIS](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf). SSOR uses EGIS to create initial regions and then uses greedy algorithm to form categorize similar groups. And with the help of color channel , Image segmentation and Classification is done. Popular architecture are using SSOR and EGIS like Region with CNN features also knows as __R-CNN__ , __Fast R-CNN__, __Faster R-CNN__ where each one the architecture remove cons of previous one respectively. Now interestingly, we can add two additional convulation layer to build __Mask R-CNN__ from __Faster R-CNN__ architecture. Both the architecture as shown below Faster R-CNN vs Mask RCNN.



 ![](https://github.com/jagatabhay/miscellaneous/blob/master/FasterRCNNArchitecture.jpg)
 
 ![](https://github.com/jagatabhay/miscellaneous/blob/master/MaskRCNNArchitecture.png)

Work link Summary :
- .
- Model : ResNet-18
- Epoch : 50
- Learning Rate :
- Batch Size :
- Highest Accuracy - 
- [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S13)

<p align>
</details>

<details>
  <summary>15. Transfer Learning</summary>
  
  this need to be updated
</details>


---


### License 

This project is licensed under the MIT license.

See [![License](https://img.shields.io/badge/LICENSE-MIT-brightgreen)](https://github.com/jagatabhay/TSAI/blob/master/LICENSE) for more details.


### Reference / Study Materials :
 - Blogs on [Medium.com](https://medium.com) and [towards data science](https://towardsdatascience.com/)
 - Research Paper on [Arxiv.org](https://arxiv.org/)
 - Andrej Karapathy lecture on Youtube - ![AK](https://github.com/jagatabhay/TSAI/blob/master/S13/logo.png)
 - Youtube Videos on Python,Pytorch.


---

### Author Info / Contributors :
- Email : 
- [![Linkedin](https://github.com/jagatabhay/TSAI/blob/master/logo.png)](https:google.com)
- [![Github](https://github.com/jagatabhay/TSAI/blob/master/S13/githublogo.png)](https://github.com/jagatabhay)
