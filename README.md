[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)


# Applied Deep Learning : Convolution Neural Network 

[![Python](https://img.shields.io/badge/Language%20%26%20Version-Python%203.6%2B-brightgreen?logo=python)](https://www.python.org/)&nbsp;&nbsp;[![PyTorch](https://img.shields.io/badge/Library-PyTorch-brightgreen?logo=pytorch)](https://pytorch.org)&nbsp;&nbsp;![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen?logo=github)&nbsp;&nbsp;[![License](https://img.shields.io/badge/LICENSE-MIT-brightgreen)](https://github.com/jagatabhay/TSAI/blob/master/LICENSE)&nbsp;&nbsp;![LAST_COMMIT](https://img.shields.io/github/last-commit/jagatabhay/TSAI)&nbsp;&nbsp;![Contriutors](https://img.shields.io/github/contributors/jagatabhay/TSAI?style=plastic)&nbsp;&nbsp;


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


Basics of CNN , how CNN learns , how different channels are formed , how DNN make sense of the inputs it gets ( __Features -> Edges & Gradients -> Patterns -> Part of Objects -> Objects__ ) . Please see below

<p align='center'>
  <img src="https://github.com/jagatabhay/miscellaneous/blob/master/Edges%20and%20Gradient.PNG">
</p>




</details>

<details>
  <summary>2. CNN Architecture </summary>
  
  Basic CNN Architecture , maintain symmetry by chosing odd size kernel(Example : 3X3 , 5X5), Importance of choosing 3X3 kernel over 5X5 or higher odd kernel , Max-Pooling  , Receptive Field.
  
  <p align='center'>
  <img src= 'https://github.com/jagatabhay/miscellaneous/blob/master/RF.gif'>
  </p>
  
 </details>
 
 <details>
  <summary>3. Kernels and Convolution </summary>
  
  This need to update
 </details>


<details>
<summary>4. Architecture Basics</summary>

[Work Link](https://github.com/jagatabhay/S4/)
</details>



<details>
<summary>5. Model Implementation</summary>

[Work Link](https://github.com/jagatabhay/S5/)
</details>



<details>
<summary>6. Batch Normalization and Regularization</summary>
  
 [Work Link](https://github.com/jagatabhay/TSAI/tree/master/S6)
</details>


<details>
<summary>7. Advanced Convolutions </summary>

[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S7)
</details>


<details>
<summary>8. Receptive Fields and dfferent Netwwork Architecture </summary>

[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S8)
</details>


<details>
<summary>9. Data Augmentation </summary>

[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S9)
</details>


<details>
<summary>10. Training and Learning Rates </summary>

[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S10)
</details>


<details>
<summary>11. Super-Convergence </summary>

[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S11)
</details>


<details>
<summary>12. Object Localization </summary>

[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S12)
</details>



<details>
<summary>13. YOLO 2 & 3 </summary>
[Work Link](https://github.com/jagatabhay/TSAI/tree/master/S13)
</details>


<details>
  <summary>14. RCNN </summary>
  
  This needs to be update.
</details>

<details>
  <summary>15. Transfer Learning</summary>
  
  this need to be updated
</details>


---


### License 

This project is licensed under the MIT license.

See [![License](https://img.shields.io/badge/LICENSE-MIT-brightgreen)](https://github.com/jagatabhay/TSAI/blob/master/LICENSE) for more details.

---


### Author Info
- Email : [jagatabhay@gmail.com](jagatabhay@gmail.com)
- [![Linkedin](https://github.com/jagatabhay/TSAI/blob/master/logo.png)](https://www.linkedin.com/in/jagatnandan-prasad-240042129/)
- [![Github](https://github.com/jagatabhay/TSAI/blob/master/S13/githublogo.png)](https://github.com/jagatabhay)
