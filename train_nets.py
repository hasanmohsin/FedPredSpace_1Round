import torch
import numpy as np

import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use('Agg')

from torchvision import datasets, transforms

##########################

lr = 1e-3
batch_size = 100
num_epochs = 10
##########################


use_cuda = torch.cuda.is_available()

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

print("CUDA available?: ", torch.cuda.is_available())
print("Device used: ", device)

# data augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

#60,000 datapoints, 28x28
train_data = datasets.MNIST(
    root = '../../data',
    train =True,
    transform = transform_train,
    download= True
)

#10,000 datapoints
val_data = datasets.MNIST(
    root = "../../data",
    train=False,
    download = True,
    transform = transform_val
)


if use_cuda:
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            num_workers=3)

else:
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False,
                                            num_workers=3)

import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, inp_dim, num_hidden, out_dim):
        super().__init__()

        self.input_dim = inp_dim
        self.num_hidden = num_hidden

        self.fc1 = nn.Linear(inp_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, out_dim)
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5, stride = 1, padding = 2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 60)
        self.fc2 = nn.Linear(60, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x #logits for classification


net = LinearNet(inp_dim=28*28, num_hidden = 100, out_dim=10)
#net = CNN()
optimizer = torch.optim.SGD(net.parameters(), lr = lr)

net = net.train()

criterion = nn.CrossEntropyLoss()

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
        
