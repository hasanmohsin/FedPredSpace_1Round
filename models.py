import torch
import torch.nn as nn
import torch.nn.functional as F

#for the neural networks trained 

class LinearNet(nn.Module):
    def __init__(self, inp_dim, num_hidden, out_dim):
        super().__init__()

        self.input_dim = inp_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim

        self.fc1 = nn.Linear(inp_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, out_dim)
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze()
        return x

#for CIFAR10, and CIFAR100
class CNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.out_dim = num_classes

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

