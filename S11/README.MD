# Session : 11                  

# Super Convergence

## Objective

Write a code which uses this new ResNet Architecture for Cifar10:

---
- PrepLayer - Conv 3x3 s1, p1 >> BN >> RELU [64K ]

- Layer 1 
   * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
   * R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
   * Add(X, R1)

- Layer 2 
   * Conv 3x3 [256k]
   * MaxPooling2D
   * BN
   * ReLU

- Layer 3 -
    * X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    * R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    * Add(X, R2)

- MaxPooling with Kernel Size 4 and Stride 2
- Fully Connected Layer 
- SoftMax

---

**Use One Cycle Policy such that:**  [Super Convergence Reference](https://arxiv.org/pdf/1708.07120.pdf)
- Total Epochs = 24
- Max at Epoch = 5
- LRMIN = FIND
- LRMAX = FIND
- NO Annihilation
 
 **Uses this transform**
 - RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
 - Batch size = 512
 - Target Accuracy: 90%.


## Results 
![LR Range Test Result Graph](https://github.com/jagatabhay/TSAI/blob/master/S11/LR-Range%20Test%20Result%20Graph.PNG)

- Maxium Learning Rate was found using LR-Range Test [Github Reference](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py)
  * Maximum LR = 0.01
- One Cycle Policy div_factor = 10
  * Initial Learning Rate = 0.01/10 = 0.001
- final_div_factor = 1 as NO Annihilation
 
 [Back To Top](#Super-Convergence)
 
 ## Files
 The IPYNB file can be found [here](https://github.com/jagatabhay/TSAI/blob/master/S11/S11_Assignment.ipynb)
 
 Source code can be found below :
  * [CustomResNet Neural Network Model](https://github.com/jagatabhay/TSAI/blob/master/S11/CustomResNet.py)
  * [Albumentation Transformation](https://github.com/jagatabhay/TSAI/blob/master/S11/albumentationstransform.py)
  * [Data Loader](https://github.com/jagatabhay/TSAI/blob/master/S11/dataloader.py)
  * [LR-Range Test](https://github.com/jagatabhay/TSAI/blob/master/S11/LRScheduler.py)
  * [Training DataSet](https://github.com/jagatabhay/TSAI/blob/master/S11/LRScheduler.py)
  * [Testing Dataset](https://github.com/jagatabhay/TSAI/blob/master/S11/traindataset.py)
  
 
 *Best Validation Accuracy : 91%  *
 

### Zig Zag Curve  One Cycle Policy Representation
   ![Zig-Zag Curve](https://github.com/jagatabhay/TSAI/blob/master/S11/S11_Figure.png)


Author Info :
- Facebook : [abhaykrishn](https://www.facebook.com/abhaykrishn)


[Back To Top](#Super-Convergence)
