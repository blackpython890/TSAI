# Session : 11                  

# Super Convergence

Assignment:

Write a code that draws Zig-ZAG/Cyclic Triangle curve (without the arrows). In submission, you'll upload your drawn curve and code for that.


Write a code which uses this new ResNet Architecture for Cifar10:

1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]

2. Layer1 -
   2.a X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
   2.b R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
   2.c Add(X, R1)

3. Layer 2 -
   3.a Conv 3x3 [256k]
   3.b MaxPooling2D
   3.c BN
   3.d ReLU
 
4.Layer 3 -
    4.a X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    4.b R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    4.c Add(X, R2)
    
5. MaxPooling with Kernel Size 4
6. FC Layer 
7. SoftMax

Uses One Cycle Policy such that:
  1.Total Epochs = 24
  2. Max at Epoch = 5
  3. LRMIN = FIND
  4. LRMAX = FIND
  5. NO Annihilation
 
 Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
 Batch size = 512
 Target Accuracy: 90%.