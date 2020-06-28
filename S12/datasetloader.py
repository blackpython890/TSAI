import torch
import torchvision.transforms as transforms
import torchvision


def datasetloader(albumentationstransform_train_transforms , batchsize , numwork ):
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    
    test_trans = transforms.Compose([ transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
                               ]) 
    
    
    traindataset  = torchvision.datasets.ImageFolder( 'TinyImageNet/train' ,
                                                        transform = albumentationstransform_train_transforms 
                                                        )
                                                        
                                        
    trainloader = torch.utils.data.DataLoader(traindatasets ,
                                              batch_size = batchsize,
                                              shuffle = True,
                                              num_workers = numwork , 
                                              pin_memory = True 
                                              )
    
    
    testset = test_datasets = datasets.ImageFolder( 'TinyImageNet/val' , 
                                                    transform = test_trans 
                                                    )
    
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size = batchsize ,
                                             shuffle = False, 
                                             num_workers = numwork , 
                                             pin_memory = True
                                             )
    
    print(" ---------------- INFO : Trainloader and Testloader Done--------------------------- ")
    return trainloader , testloader