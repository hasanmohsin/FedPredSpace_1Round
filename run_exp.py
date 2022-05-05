import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import matplotlib

import datasets
import models
import train_nets
import utils

#device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def main(args):

    ####################     
    lr = 1e-3
    batch_size = 100
    num_epochs = 2
    ####################

    utils.set_seed(args.seed)

    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    print("CUDA available?: ", torch.cuda.is_available())
    print("Device used: ", device)

    trainloader, valloader  = datasets.get_mnist(use_cuda, batch_size)

    net = models.LinearNet(inp_dim=28*28, num_hidden = 100, out_dim=10)
    
    net = train_nets.sgd_train(net, lr, num_epochs, trainloader)
    
    acc = utils.classify_acc(net, valloader)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12)
    
    #dataset stuff
    parser.add_argument('--batch_size', type=int, default = 100)

    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--num_epochs', type=int, default = 5)

    parser.add_argument('--save_dir', type=str, default = "./plots/")

    #for later - setting up train/test split
    parser.add_argument('--ntrain', type=int, default=500)
    parser.add_argument('--train_percent', type=float, default=0.8)


    args = parser.parse_args()
    
    main(args)
