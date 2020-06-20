# Session : 11                  

# Super Convergence

## Objective :

Write a code which uses this new ResNet Architecture for Cifar10:

- PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64K ]

- Layer1 
   X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
   R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
   Add(X, R1)


- Layer 2 -
   3.a Conv 3x3 [256k]
   3.b MaxPooling2D
   3.c BN
   3.d ReLU


- Layer 3 -
    4.a X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    4.b R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    4.c Add(X, R2)


- MaxPooling with Kernel Size 4 and Stride 2
- FC Layer 
- SoftMax

Uses One Cycle Policy such that:
- Total Epochs = 24
- Max at Epoch = 5
- LRMIN = FIND
- LRMAX = FIND
- NO Annihilation
 
 Uses this transform
 - RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
 - Batch size = 512
 - Target Accuracy: 90%.


[Back To Top](#Super-Convergence)
