from pkgutil import get_data
import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import tensorflow as tf

#do an noniid split of the dataset into num_clients parts
# each client gets equal sized dataset specified by the client_data_size
def non_iid_mnist_split(dataset, num_clients, client_data_size, batch_size, shuffle, shuffle_digits=True):
    assert(num_clients>0 and num_clients<=10)

    digits=torch.arange(10) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))

    # split the digits in a fair way
    digits_split=list()
    i=0
    for n in range(num_clients, 0, -1):
        inc=int((10-i)/n)
        digits_split.append(digits[i:i+inc])
        i+=inc

    # load and shuffle num_clients*client_data_size from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=num_clients*client_data_size,
                                        shuffle=shuffle)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = dataiter.next()

    data_splitted=list()
    for i in range(num_clients):
        idx=torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool() # get indices for the digits
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size, shuffle=shuffle))

    return data_splitted

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


## Bike Sharing dataset
# 511 training points
# 220 validation points
def get_bike(normalize = True, batch_size = 1):
    col1=['instant','dteday','season','yr','mnth','holiday','weekday',
     'workingday','weathersit','temp','atemp','hum','windspeed','casual','registered', 'cnt']
    df1 = pd.read_csv('Dataset/bike.csv',header=None,skiprows=1, na_filter=True,names=col1)
    df1.dropna()
    df1 = df1.drop(columns=['dteday'])
    df1 = df1.drop(columns=['instant'])
    col1=df1.columns.tolist()
    if normalize:
        df1 =(df1-df1.min())/(df1.max()-df1.min())
    data=df1[col1]
    target = data.pop('cnt')
    ds = tf.data.Dataset.from_tensor_slices((data.values, target.values))
    train_size = int(len(ds) * 0.7)
    dataset = (
        ds
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=len(ds))
        .cache()
    )
    # We shuffle with a buffer the same size as the dataset.
    train_dataset = DataLoader(
        dataset.take(train_size), batch_size, shuffle = True
    )
    test_dataset = DataLoader(
        dataset.skip(train_size), batch_size, shuffle = True
    )
    return train_dataset, test_dataset


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
