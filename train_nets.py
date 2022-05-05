import torch
import numpy as np

import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use('Agg')

import datasets
import models

##########################

lr = 1e-3
batch_size = 100
num_epochs = 10
##########################


use_cuda = torch.cuda.is_available()

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

print("CUDA available?: ", torch.cuda.is_available())
print("Device used: ", device)

trainloader, valloader  = datasets.get_mnist(use_cuda, batch_size)

net = models.LinearNet(inp_dim=28*28, num_hidden = 100, out_dim=10)
#net = CNN()
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

# test eval
total = 0
correct = 0

net = net.eval()
for x,y in valloader:
    pred_logit = net(x)
    _, pred = torch.max(pred_logit, 1)    

    total += y.size(0)
    correct += (pred == y).sum().item()

print("Accuracy on test set: ", 100*correct/total)
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
        
