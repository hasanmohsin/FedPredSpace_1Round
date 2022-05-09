from pkgutil import get_data
import torch
from torchvision import datasets, transforms
import numpy as np

#do an iid split of the dataset into num_clients parts
# each client gets equal sized dataset
def iid_split(data, num_clients, batch_size):
    data_size = data.targets.shape[0]
    c_data_size = int(np.floor(data_size/num_clients))

    #to use all data
    c_data_size_last = data_size - c_data_size*(num_clients - 1)

    lens = num_clients*[c_data_size]
    lens[-1] = c_data_size_last

    c_data = torch.utils.data.random_split(data, lens)

    c_dataloaders = []

    #construct dataloaders
    for shard in c_data:
        c_dataloader = torch.utils.data.DataLoader(shard, 
                                                batch_size=batch_size, shuffle=True, 
                                                pin_memory=True)
        c_dataloaders.append(c_dataloader)

    #array of datasets
    return c_dataloaders, lens

## MNIST dataset
# 60,000 train points, 
# 10,000 validation points
def get_mnist(use_cuda, batch_size, get_datamat = False):
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
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True
                                                , num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True
                                                ,num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False
                                                ,num_workers=3)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False
                                             ,num_workers=3)
    if get_datamat:
        return trainloader, valloader, train_data
    else:
        return trainloader, valloader
