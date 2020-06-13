import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.preplayer = nn.Sequential( nn.Conv2d( 3, 64, 3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU() )
                                    
        self.layer1x = nn.Sequential( nn.Conv2d( 64, 128, 3, padding=1),
                                      nn.MaxPool2d( 2, 2),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU() )
                                    
        self.layer1R1 = nn.Sequential( nn.Conv2d( 128, 128, 3, padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Conv2d( 128, 128, 3, padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU() )
        
        self.layer2 = nn.Sequential( nn.Conv2d( 128, 256, 3, padding=1),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU() )

        self.layer3x = nn.Sequential( nn.Conv2d( 256, 512, 3, padding=1),
                                     nn.MaxPool2d(2,2),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU() )

        self.layer2R2 = nn.Sequential ( nn.Conv2d( 512, 512, 3, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.Conv2d(512,512,3,padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU() )
        self.pool =  nn.MaxPool2d(4,1)
        self.end =  nn.Sequential(nn.Linear(512,10)) 


    def forward(self, x):
        x = self.preplayer(x)
        x = self.layer1x(x)
        residue1 = self.layer1R1(x)
        x = x + residue1
        x = self.layer2(x)
        x = self.layer3x(x)
        residue2 = self.layer2R2(x)
        x = x + residue2
        x = self.pool(x)
        x = x.view(-1, 512 )
        x = self.end(x) 
        return F.log_softmax(x)