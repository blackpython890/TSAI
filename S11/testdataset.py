import torch


def testdataset(model , device , testloader ):
    correct = 0
    total = 0
    with torch.no_grad():
          for data in testloader:
               images , labels = data[0].to(device) , data[1].to(device)
               output = model(images)
               _ , predicted = torch.max(output.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
     
     
     
    print('Accuracy of the Network on the 10000 test images: %d %%' % (100 * correct / total))