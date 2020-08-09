:star:

# Monocular Depth Estimation and Image Segmentation
[![Python](https://img.shields.io/badge/Language%20%26%20Version-Python%203.6%2B-brightgreen)](https://www.python.org/)&nbsp;&nbsp;[![PyTorch](https://img.shields.io/badge/Library-PyTorch/OpenCV2-brightgreen)](https://pytorch.org)&nbsp;&nbsp;[![License](https://img.shields.io/badge/LICENSE-MIT-brightgreen)](https://github.com/jagatabhay/TSAI/blob/master/LICENSE)&nbsp;&nbsp;![Contributors](https://img.shields.io/github/contributors/jagatabhay/TSAI?style=plastic)&nbsp;&nbsp;[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)&nbsp;&nbsp;


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

**Create Custom Dataset**

- [Background Images](#background-images)
   - Image Type : forest scenery without animals downloaded from internet and then resized
   - Image Dimension : (224 , 224 , 3).
   - Total Images : 100
   - Dataset Sample shown below
   - ![A](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg001.jpg)&nbsp;![B](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg013.jpg)&nbsp;![C](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg025.jpg)&nbsp;![D](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg035.jpg)&nbsp;![E](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg050.jpg)&nbsp;![F](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg052.jpg)&nbsp;![G](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg073.jpg)&nbsp;![H](https://github.com/jagatabhay/TSAI/blob/master/S15/background/bg080.jpg)


- [Foreground Images](#forground-images)
  - Image Type : Animals with white background downloaded from internet and then resized
  - Image Dimension : (125 , 125 , 3)
  - Total Images : 100
  - Dataset Sample shown below
  - ![A](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg001.jpg)![B](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg005.jpg)![C](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg015.jpg)![D](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg020.jpg)![E](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg024.jpg)![F](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg036.jpg)![G](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg046.jpg)![H](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg053.jpg)![I](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground/fg061.jpg)

- [Foreground Mask Images](#foreground-mask-images)
   - Image Type : Mask Images of for forground image.
   - Image Dimension : (125 , 125 , 3)
   - Total Images : 100
   - Below Source code is used to form mask images.
   - ![Images](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/carbon.png)
   - Dataset sample shown below
   - ![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms021.jpg)&nbsp;![b](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms027.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms033.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms058.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms062.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms069.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms074.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms078.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/foreground%20masks/ms084.jpg)&nbsp;
   - Colab File - [![Colab](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/InvertImage.ipynb)
 
 - [Overlay Images](#overlay-images)
   - Image Type : Randomly place the foreground on the background 20 times. 100 Bacground + 200 Foreground(100 Fg + 100 fg flip)
   - Image Dimension : (224 , 224 , 3)
   - Below source code is used to form overlay images.
   - Total Images : 400K
   - ![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/overlayImgSourceCode.png)
   - Sample Dataset is shown below
   - ![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg105190.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg125460.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg156810.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg82.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg66125.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg258129.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg229170.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/overlayImages/fg_bg193472.jpg)&nbsp;
   - Colab File - [![Colab](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/fg_bg.ipynb)
   
   
 - [Overlay Mask Images](#overlay-mask-images)
   - Image Type : Mask Images of [overlay images](#overlay-images)
   - Total Images : 400K
   - Image Dimension : (224 , 224 , 3)
   - Tool used : [GIMP](https://www.gimp.org/)
   - Sample Dataset shown below
   - ![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg100051.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg124052.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg136056.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg184068.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg42001.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg74035.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg88042.jpg)&nbsp;![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_overlaymask/fg_bg90046.jpg)&nbsp;



- [Depth Images](#depth-images)
   - Image Type : Depth Images of [overlay images](#overlay-images)
     - As we don't have Depth cameras , trained model is used to generate depth images. This depth images is used as ground truth while training model.
     - Github Reference : [![a](https://github.com/jagatabhay/TSAI/blob/master/S13/githublogo.png)](https://github.com/priya-dwivedi/Deep-Learning)
   - Total Images - 400K
   - Image Dimension : (224 , 224 , 3)
   - Sample Dataset shown below
   - ![a](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg16009.jpg)&nbsp;![b](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg16162.jpg)&nbsp;![b](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg200.jpg)&nbsp;![b](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg4057.jpg)&nbsp;![b](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg50.jpg)&nbsp;
![b](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg5507.jpg)&nbsp;
![b](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg8126.jpg)&nbsp;
![b](https://github.com/jagatabhay/TSAI/blob/master/S15/groundtruth_depthmages/fg_bg8407.jpg)&nbsp;


---

### DataSets Size and Links

| Seril No | Datasets Name | Datasets Size | Total Images | Dataset Link Google Drive |
| -------- | ------------ | -------------- | ------------ | ----------------- |
| 1 | Background Image |  4.67 MB | 100 | [Google Drive](https://drive.google.com/drive/folders/1plfyAcoQm6BM6xwXn0vTWWda84qur6VU?usp=sharing) |
| 2 | Foreground Image | 512 KB | 200 | [Google Drive](https://drive.google.com/drive/folders/1JDmJYgzkoyvdA79zQfoTZ1JACKIzbcdQ?usp=sharing) |
| 3 | Foreground Mask Images | 0.912 MB | 200 | [Google Drive](https://drive.google.com/drive/folders/1PZoZ19E523IfehC_Xw-CGywsxEuaRq-o?usp=sharing) |
| 4 | Overlay Images | 14 GB | 400K | [Google Drive Set-1](https://drive.google.com/drive/folders/1vPEjYFLWSt6PdI4MEpcsjiU04aqyO3g3?usp=sharing) [Google Drive Set-2](https://drive.google.com/drive/folders/1SXZ-U16ciWNMm1zlF_skTyEbhX89kKwR) |
| 5 | Overlay Mask Images | - | 400K | - |
| 6 | Depth Images | Please Update | 400K | [Google Drive Set-1](https://drive.google.com/drive/folders/15lSuGG3oyg3A3_SP_N6CXHi1iFVtL74J?usp=sharing) [Google Drive Set-2](https://drive.google.com/drive/folders/1oc1hO56bwnFgg9F1b3Z32OgFXDaPNJXi) |

**Note :**
  - Drive Set-1 Contains 276000 images
  - Drive Set-2 Contains 124000 images

---

### Dataset Statistics
- Colab File - [![ColabFile1](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/DatasetStats_bg.ipynb)&nbsp;&nbsp;[![ColabFile2](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/DatasetStats_fg.ipynb)&nbsp;&nbsp;[![ColabFile3](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/DatasetStats_fgmask.ipynb)&nbsp;&nbsp;[![ColabFile4](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/DatasetStats_fgbg.ipynb)&nbsp;&nbsp;[![ColabFile5](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/DatasetStats_fgbgDepth.ipynb)&nbsp;&nbsp;[![ColabFile6](https://github.com/jagatabhay/TSAI/blob/master/openincolablogo.JPG)](https://github.com/jagatabhay/TSAI/blob/master/S15/DatasetStats_fgbgmask.ipynb)&nbsp;&nbsp;

Source Code is mentioned below

``` Python
import torch
import torchvision

dataset = torchvision.datasets.ImageFolder('gdrive/My Drive/root',
                                           transform = torchvision.transforms.ToTensor() 
                                           )

dataloader = torch.utils.data.DataLoader(dataset ,
                                         batch_size = 1,
                                         shuffle = True,
                                         num_workers = 1 , 
                                         pin_memory = True 
                                        )
images , _ = iter(dataloader).next()

def get_mean_and_std(image):
  mean = torch.zeros(3)
  std = torch.zeros(3)
  for i in range(3):
      mean[i] += image[:, i, :, :].mean()
      std[i] += image[:,i, :, :].std()
  return mean , std 
  
mean , std = get_mean_and_std(images)
print('Mean : ',mean)
print('Std : ',std)

```
**Graphical Representation of all dataset mean and standard deviation**


![](https://github.com/jagatabhay/TSAI/blob/master/S15/DatasetsCharts.JPG)

**Note :**
 - Respective Dataset statistics data ( mean and standard deviation ) is shared in respective colab.

---

### Model

 U-Net Architecture Model is used to train on the data set.
 In the UNET architecture downsampling ( encoder ) DENSENET-169 is used and Decoder ( Upsampling )  Images is croped from the densenet block and then concatenated as mentioned in the original paper.
 
 
 ### Training 
 
 Parameters :
 - Epoech : 20
 - Batch Size : 4
 - Trained Images : 20K
 - Learning Rate : 0.0001
 
 
 ``` Python
 !python train.py --data nyu --bs 4 --full --dnetVersion small
 
 ```
 
 Logs Data
 ``` logs
 
 Epoch 1/20
250/250 [==============================] - 332s 1s/step - loss: 0.2110 - val_loss: 18.7512
Epoch 2/20
250/250 [==============================] - 268s 1s/step - loss: 0.1611 - val_loss: 18.7269
Epoch 3/20
250/250 [==============================] - 267s 1s/step - loss: 0.1476 - val_loss: 18.7756
Epoch 4/20
250/250 [==============================] - 269s 1s/step - loss: 0.1373 - val_loss: 18.7424
Epoch 5/20
250/250 [==============================] - 268s 1s/step - loss: 0.1285 - val_loss: 18.7604
Epoch 6/20
250/250 [==============================] - 268s 1s/step - loss: 0.1210 - val_loss: 18.7255
Epoch 7/20
250/250 [==============================] - 268s 1s/step - loss: 0.1162 - val_loss: 18.7370
Epoch 8/20
250/250 [==============================] - 267s 1s/step - loss: 0.1126 - val_loss: 18.7351
Epoch 9/20
250/250 [==============================] - 268s 1s/step - loss: 0.1067 - val_loss: 18.7834
Epoch 10/20
250/250 [==============================] - 267s 1s/step - loss: 0.1028 - val_loss: 18.7484
Epoch 11/20
250/250 [==============================] - 267s 1s/step - loss: 0.0986 - val_loss: 18.7541
Epoch 12/20
250/250 [==============================] - 268s 1s/step - loss: 0.0958 - val_loss: 18.7167
Epoch 13/20
250/250 [==============================] - 266s 1s/step - loss: 0.0936 - val_loss: 18.7412
Epoch 14/20
250/250 [==============================] - 267s 1s/step - loss: 0.0919 - val_loss: 18.7488
Epoch 15/20
250/250 [==============================] - 268s 1s/step - loss: 0.0900 - val_loss: 18.7129
Epoch 16/20
250/250 [==============================] - 267s 1s/step - loss: 0.0893 - val_loss: 18.7268
Epoch 17/20
250/250 [==============================] - 267s 1s/step - loss: 0.0851 - val_loss: 18.7367
Epoch 18/20
250/250 [==============================] - 268s 1s/step - loss: 0.0828 - val_loss: 18.7249
Epoch 19/20
250/250 [==============================] - 267s 1s/step - loss: 0.0820 - val_loss: 18.7354
Epoch 20/20
250/250 [==============================] - 266s 1s/step - loss: 0.0803 - val_loss: 18.7336
 
 
 ```


 
### Author Info
- Email : jagatabhay@gmail.com   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [![Linkedin](https://github.com/jagatabhay/TSAI/blob/master/logo.png)](https://www.linkedin.com/in/jagatnandan-prasad-240042129/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  [![Github](https://github.com/jagatabhay/TSAI/blob/master/S13/githublogo.png)](https://github.com/jagatabhay)
