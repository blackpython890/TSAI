# Session : 11                  

# Super Convergence

## Objective :

Write a code which uses this new ResNet Architecture for Cifar10:

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

**Uses One Cycle Policy such that:**
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
- Maxium Learning Rate was found using LR-Range Test [Reference](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py)
  * Maximum LR = 0.01
- One Cycle Policy 

Author Info :
- Facebook -- [abhaykrishn](https://www.facebook.com/abhaykrishn)



[Back To Top](#Super-Convergence)



