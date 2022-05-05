import torch
import numpy as np

def sgd_train(net, lr, num_epochs, trainloader):
    optimizer = torch.optim.SGD(net.parameters(), lr = lr)

    net = net.train()

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for x, y in trainloader:
            
            optimizer.zero_grad()
            pred_logits = net(x)
            loss = criterion(pred_logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            count+=1

        
        
        print("Epoch: ", i+1, "Loss: ", epoch_loss)
    print("Training Done!")
    return net



#def sgld_run(net, num_samples, burn_in_epochs, mix_epochs, num_samples, thin_epochs, lr):

    #params:
    #lr = 1e-3
    #thin_epochs = 100
    #num_samples = 50
    #mix_epochs = 100
    #burn_in_epochs = 1000
    

    #opt = torch.optim.SGD(net.parameters(), lr = lr)

    #num_epochs = burn_in_epochs + mix_epochs*num_samples

    #for i in num_epochs:
        
