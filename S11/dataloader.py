import torch
import torchvision.transforms as transforms
import torchvision


def datasetloader(albumentationstransform_train_transforms , batchsize , numwork ):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    trans = transforms.Compose([ transforms.ToTensor(),
                             transforms.Normalize(mean, std)
                            ]) 
    
    trainset = torchvision.datasets.CIFAR10(root = './data', 
                                            train = True ,
                                            download = True , 
                                            transform = albumentationstransform_train_transforms )
                                        
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size = batchsize,
                                              shuffle = True,
                                              num_workers = numwork , 
                                              pin_memory = True )
    
    
    testset = torchvision.datasets.CIFAR10(root='./data', 
                                           train = False,
                                           download = True, 
                                           transform = trans )
                                       
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size = batchsize ,
                                             shuffle = False, 
                                             num_workers = numwork , 
                                             pin_memory = True)
    
    print("INFO : Trainloader and Testloader Done")
    return trainloader , testloader