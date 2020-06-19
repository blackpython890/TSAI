from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip, Resize, Rotate , Cutout , RandomCrop , PadIfNeeded
from albumentations.pytorch import ToTensor
import numpy as np


class train_transforms():

    def __init__(self):
        self.albTrainTransforms = Compose([  # Resize(256, 256),
            Rotate((-10.0, 10.0)),
            HorizontalFlip(p=0.5),
            PadIfNeeded( min_height = 4 , min_width = 4 ),
            RandomCrop( height = 32 , width = 32 , p = 1.0 ) ,
            Cutout(num_holes = 1, max_h_size = 8, max_w_size = 8),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensor()
        ])
        
        # this is train transforms


    print("REQUIRED LIBRARIES LOADED...")

    def __call__(self, img):
        img = np.array( img )
        img = self.albTrainTransforms(image=img)['image']
        return img