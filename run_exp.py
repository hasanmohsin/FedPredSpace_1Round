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
    lr = args.lr #1e-3
    #batch_size = 100
    #num_epochs = 24
    
    #mode = "fed_pa"
    #mode = "fed_sgd"
    mode = args.mode #"f_mcmc"
    #mode = "global_bayes"
    ####################

    utils.set_seed(args.seed)

    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    print("CUDA available?: ", torch.cuda.is_available())
    print("Device used: ", device)

    ################################
    # DATASET
    ################################
    task = "classify"

    #MNIST default
    inp_dim = 28*28
    
    if args.dataset == "mnist":
        trainloader, valloader, train_data  = datasets.get_mnist(use_cuda, args.batch_size, get_datamat=True)
    elif args.dataset == "cifar10":
        trainloader, valloader, train_data  = datasets.get_cifar10(use_cuda, args.batch_size, get_datamat=True)
    elif args.dataset == "cifar100":
        trainloader, valloader, train_data  = datasets.get_cifar100(use_cuda, args.batch_size, get_datamat=True)
    elif args.dataset == "emnist":
        trainloader, valloader, train_data  = datasets.get_emnist(use_cuda, args.batch_size, get_datamat=True)
    elif args.dataset == "f_mnist":
        trainloader, valloader, train_data  = datasets.get_fashion_mnist(use_cuda, args.batch_size, get_datamat=True)
    
    #regression datasets
    elif args.dataset == "bike":
        task = "regression"
        trainloader, valloader, train_data = datasets.get_bike(batch_size = args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])
    elif args.dataset == "airquality":
        task = "regression"
        trainloader, valloader, train_data = datasets.get_airquality(batch_size=args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])

    if task == "classify":
        out_dim = len(train_data.classes)
    #set up network
    base_net = models.LinearNet(inp_dim=inp_dim, num_hidden = 100, out_dim=out_dim)

    ################################
    # TRAINING ALGORITHMS
    ################################
    if mode == "sgd":
        
        base_net = train_nets.sgd_train(base_net, lr, args.num_epochs_per_client*args.num_rounds, trainloader)
        
        acc = utils.classify_acc(base_net, valloader)
    elif mode == "fed_sgd":
       
        fed_avg_trainer = fed_algos.FedAvg(num_clients = args.num_clients, 
                                        base_net = base_net, 
                                        traindata = train_data, 
                                        num_rounds = args.num_rounds, 
                                        epoch_per_client = args.num_epochs_per_client,
                                        batch_size = args.batch_size, 
                                        non_iid = args.non_iid,
                                        task = task)
        fed_avg_trainer.train(valloader)
        acc = utils.classify_acc(fed_avg_trainer.global_net, valloader)
    elif mode == "fed_pa":

        fed_pa = fed_algos.FedPA(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data,
                                    num_rounds = 4,#1,
                                    epoch_per_client = 6,
                                    batch_size = args.batch_size, device=device,
                                    non_iid = args.non_iid,
                                    task = task)
        fed_pa.train(valloader=valloader)
    elif mode == "ep_mcmc":
        ep_mcmc = fed_algos.EP_MCMC(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data,
                                    num_rounds = 1,
                                    epoch_per_client = 24,
                                    batch_size = args.batch_size, device=device,
                                    non_iid = args.non_iid,
                                    task = task)
        ep_mcmc.train(valloader=valloader)
    
    elif mode == "f_mcmc":
        f_mcmc = fed_algos.F_MCMC(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data,
                                    num_rounds = 1,
                                    epoch_per_client = 24,
                                    batch_size = args.batch_size, device=device,
                                    non_iid = args.non_iid,
                                    task = task)
        f_mcmc.train(valloader=valloader)

    elif mode == "global_bayes":
        print("cSGHMC inference")
        
        trainer = train_nets.cSGHMC(base_net=base_net, trainloader=trainloader, device=device)
        trainer.train()
        acc = trainer.test_acc(valloader)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12)
    
    parser.add_argument('--dataset', type= str, default = "MNIST")
    parser.add_argument('--non_iid', action="store_true") 

    parser.add_argument('--mode', type=str, default = "fed_sgd")

    #dataset stuff
    parser.add_argument('--batch_size', type=int, default = 100)

    parser.add_argument('--lr', type=float, default=1e-3)

    #for federated learning
    parser.add_argument('--num_rounds', type=int, default = 6)
    parser.add_argument('--num_epochs_per_client', type=int, default = 4)

    parser.add_argument('--num_clients', type = int, default = 5)


    parser.add_argument('--save_dir', type=str, default = "./plots/")

    #for later - setting up train/test split
    #parser.add_argument('--ntrain', type=int, default=500)
    #parser.add_argument('--train_percent', type=float, default=0.8)

    args = parser.parse_args()
    
    main(args)
