from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip, Resize,Rotate , Cutout
from albumentations.pytorch import ToTensor
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import numpy as np


class train_transforms():

    def __init__(self):
        self.albTrainTransforms = Compose([  # Resize(256, 256),
            Rotate((-10.0, 10.0)),
            HorizontalFlip(p=0.5),
            #VerticalFlip(p=0.5),
            #transforms.RandomCrop(size = [32,32], padding = 4),
            Cutout(num_holes = 1, max_h_size = 8, max_w_size = 8),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensor()
        ])# this is train transforms

        self.tran = transforms.Compose([transforms.RandomCrop(size=[32,32] , padding = 4 )])
        self.aug = iaa.Fliplr(0.5)

    print("REQUIRED LIBRARIES LOADED...")

    def __call__(self, img):
        img = np.array( img )
        #img = self.tran(image = img )
        img = self.albTrainTransforms(image=img)['image']
        return img
