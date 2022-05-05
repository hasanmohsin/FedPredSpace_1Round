import torch
from torchvision import datasets, transforms

## MNIST dataset
# 60,000 train points, 
# 10,000 validation points
def get_mnist(use_cuda, batch_size):
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

    return trainloader, valloader
