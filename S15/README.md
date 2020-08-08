
# RCNN and Monocular Depth Estimation
[![Python](https://img.shields.io/badge/Language%20%26%20Version-Python%203.6%2B-brightgreen)](https://www.python.org/)&nbsp;&nbsp;[![PyTorch](https://img.shields.io/badge/Library-PyTorch/OpenCV2-brightgreen)](https://pytorch.org)&nbsp;&nbsp;[![License](https://img.shields.io/badge/LICENSE-MIT-brightgreen)](https://github.com/jagatabhay/TSAI/blob/master/LICENSE)&nbsp;&nbsp;

## Objective

Build your own custom dataset and use this dataset for monocular depth estmation and segmentation simultaneously.

You must have following images :
 * 100 background ( Scenes like fronts of the shops , malls , airport etc ) of you own choice.
 * 100 foreground and 100 including mask of respective foreground. Total 200 of own choice.


Steps need to be followed for creating custom dataset
 * Select 100 images of background
 * Select 100 images of object with transparent background.
 * Build 100 Mask of the images with transparent background.
 * Overlay foreground on the top of background and mask of the background. This images will be called as fg_bg.\
   Randomly place the foreground on the background 20 times and in total 100x200x20 images formed 
 * Create equivalent mask of the fg_bg images.

Use this fg_bg produced dataset in Monocular Depth Model and generate **Monocular Depth Maps**

<p align="center">
   <img width="400" height="400" src="https://2.bp.blogspot.com/-x8Ft8PeU5t4/W_2GXlXjYqI/AAAAAAAADi4/-h__RwPtD4Y9WcjfiOMlCuyTpTkwK6m1gCLcBGAs/s1600/image7.png">
</p> 


**Total Images Tally**
* 400K fg_bg Images
* 400K Depth Images
* 400K Depth Mask Images

***Total Images : 1.2 Million***


Points to consider
1. Go for ***Square Images***  as it helps.
1. Use above images in a network which would take an fg_bg image AND bg image, and predict your MASK and Depth image.\
   So the input to the network is, say, 224x224xM and 224x224xN, and the output is 224x224xO and 224x224xP.
1. Pick the Resolution of your own choice.


---

[Back to Top](#rcnn-and-monocular-depth-estimation)

## Solution

Create Custom Dataset

- [Background Images](#background-images)
   - Image Type : forest scenery without animals.
   - Image size : 224 * 224
   - Total Images : 100
   - Dataset Sample shown below
   - ![A](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg001.jpg)&nbsp;![B](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg013.jpg)&nbsp;![C](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg025.jpg)&nbsp;![D](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg035.jpg)&nbsp;![E](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg050.jpg)&nbsp;![F](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg052.jpg)&nbsp;![G](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg073.jpg)&nbsp;![H](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg080.jpg)


   
- [Foreground Images](#forground-images)
  - Image Type : Animals with white background.
  - Image size : 125 * 125
  - Total Images : 100
  - Dataset Sample shown below
  - ![A](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg001.jpg)![B](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg005.jpg)![C](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg015.jpg)![D](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg020.jpg)![E](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg024.jpg)![F](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg036.jpg)![G](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg046.jpg)![H](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg053.jpg)![I](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg061.jpg)

- [Foreground Mask Images](#foreground-mask-images)
   - Image Type : Mask Images of for forground image.
   - Image Size : 125 * 125
   - Total Images : 100
   - 
