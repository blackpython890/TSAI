loss = []

def traindataset(range_ , model , device , trainloader , optimizer , criterion_ , batchsize , scheduler_ = None ):
     for epoch in range(range_):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device),data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion_(outputs, labels)
            loss.backward()
            
            optimizer.step()
            if scheduler_ is not None:
                    scheduler_.step()
            
          
            # print statistics
            running_loss += loss.item()
            minibatch = 100000//batchsize
            if i % minibatch == minibatch-1 :    # print every 2000 mini-batches
               print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / minibatch-1 ))
               loss.append(running_loss / minibatch-1)
               running_loss = 0.0

    
     print('INFO : Finished Training of Dataset ')
     return loss     
