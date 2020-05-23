Sesssion : S8
Objective : To Achieve 85% accuracy by using ResNet-18 Model.
Dataset : CIFAR-10.

Srategy : 
Dataset is transformed with different Image Augmentation.
Image Augmentation:
1. RandomHorizontalFlip(p=0.5)
2. RandomCrop(32, padding=2)
3. ColorJitter(brightness=0.1, contrast=0.1,saturation=0.1,hue=0.1) .

And Then Normalize.

Number Of Parameters : 11,173,962 ( Trainable parameters ) .


Result Achieved:
Accuracy : 87%
Epoch : 20
Classwise Accuracy
1. Highlest Accuracy : Ship , Frog .
2. Lowest Accuracy : Cat .



To Load and Run our Model .
Kindly run the File : S8-Assignment.ipynb  in google Colab and when asked upload the RestNetModel.py
