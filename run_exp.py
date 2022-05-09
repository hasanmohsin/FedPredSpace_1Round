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
import fed_algos

#device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def main(args):

    ####################     
    lr = 1e-3
    batch_size = 100
    num_epochs = 24
    mode = "EP_MCMC"
    #mode = "cSGHMC"
    ####################

    utils.set_seed(args.seed)

    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    print("CUDA available?: ", torch.cuda.is_available())
    print("Device used: ", device)

    trainloader, valloader, train_data  = datasets.get_mnist(use_cuda, batch_size, get_datamat=True)

    if mode == "SGD":
        net = models.LinearNet(inp_dim=28*28, num_hidden = 100, out_dim=10)
        
        net = train_nets.sgd_train(net, lr, num_epochs, trainloader)
        
        acc = utils.classify_acc(net, valloader)
    elif mode == "fed_SGD":
        base_net = models.LinearNet(inp_dim = 28*28, num_hidden = 100, out_dim = 10)

        fed_avg_trainer = fed_algos.FedAvg(num_clients = 5, 
                                        base_net = base_net, 
                                        traindata = train_data, 
                                        num_rounds = 10, 
                                        epoch_per_client = 2,
                                        batch_size = batch_size)
        fed_avg_trainer.train(valloader)
        acc = utils.classify_acc(fed_avg_trainer.global_net, valloader)

    elif mode == "EP_MCMC":
        base_net = models.LinearNet(inp_dim = 28*28, num_hidden = 100, out_dim = 10)

        ep_mcmc = fed_algos.EP_MCMC(num_clients = 5,
                                    base_net = base_net,
                                    traindata=train_data,
                                    num_rounds = 1,
                                    epoch_per_client = 24,
                                    batch_size = batch_size, device=device)
        ep_mcmc.train(valloader=valloader)

    elif mode == "cSGHMC":
        print("cSGHMC inference")

        net = models.LinearNet(inp_dim = 28*28, num_hidden  =100, out_dim = 10)
        
        trainer = train_nets.cSGHMC(base_net=net, trainloader=trainloader, device=device)
        trainer.train()
        acc = trainer.test_acc(valloader)

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
