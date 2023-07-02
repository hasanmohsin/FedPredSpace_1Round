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
import nvp

#device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def main(args):

    utils.makedirs(args.save_dir)

    if args.mode == "fed_sgd":
        exp_id = "{}_{}_{}_clients_{}_rounds_{}_optim_log_{}_noniid".format(args.dataset, args.mode, args.num_clients, args.num_rounds, args.optim_type, args.non_iid)    
        fname = "{}/{}_{}".format(args.save_dir, exp_id, args.seed) 
    else:
        #took out seed name for non-fed-sgd runs so that the results are in a single dict - this is a hack - change later!
        exp_id = "{}_{}_{}_clients_{}_rounds_log_{}_noniid".format(args.dataset, args.mode, args.num_clients, args.num_rounds, args.non_iid)
        fname = "{}/{}".format(args.save_dir, exp_id)
        
    model_save_dir = "{}/models".format(args.save_dir)    
    

    logger = open(fname+".txt", 'w')

    utils.print_and_log("Experiment: Args {}".format(args), logger)

    ####################     
    #lr = args.lr #1e-3
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
        inp_dim = 28*28
    elif args.dataset == "cifar10":
        trainloader, valloader, train_data  = datasets.get_cifar10(use_cuda, args.batch_size, get_datamat=True)
        inp_dim = 32*32*3
    elif args.dataset == "cifar100":
        trainloader, valloader, train_data  = datasets.get_cifar100(use_cuda, args.batch_size, get_datamat=True)
        inp_dim = 32*32*3
    elif args.dataset == "emnist":
        trainloader, valloader, train_data  = datasets.get_emnist(use_cuda, args.batch_size, get_datamat=True)
        inp_dim = 28*28
    elif args.dataset == "f_mnist":
        trainloader, valloader, train_data  = datasets.get_fashion_mnist(use_cuda, args.batch_size, get_datamat=True)
        inp_dim = 28*28

    #regression datasets
    elif args.dataset == "bike":
        task = "regression"
        trainloader, valloader, train_data, df = datasets.get_bike(batch_size = args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])
    elif args.dataset == "airquality":
        task = "regression"
        trainloader, valloader, train_data, df = datasets.get_airquality(batch_size=args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])
    elif args.dataset == "forest_fire":
        task = "regression"
        trainloader, valloader, train_data, df = datasets.get_forestfire(batch_size=args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])
    elif args.dataset == "real_estate":
        task = "regression"
        trainloader, valloader, train_data, df = datasets.get_realestate(batch_size=args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])    
    elif args.dataset == "winequality":
        task = "regression"
        trainloader, valloader, train_data, df = datasets.get_winequality(batch_size=args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])
    elif args.dataset == "power":
        task = "regression"
        trainloader, valloader, train_data, df = datasets.get_power(batch_size=args.batch_size)
        out_dim = 1
        inp_dim = len(train_data[0][0])

    if task == "classify":
        out_dim = len(train_data.classes)
    
    if args.net_type == "fc":
        #set up network
        base_net = models.LinearNet(inp_dim=inp_dim, num_hidden = 100, out_dim=out_dim)
        base_net = base_net.to(device)

    elif args.net_type == "cnn":
        base_net = models.CNN(num_classes=out_dim)
        base_net = base_net.to(device)


    ###################################
    # Hyperparams
    ###################################
    #for sgd technique (per client)
    sgd_hyperparams = { 'epoch_per_client': args.epoch_per_client,
                        'lr': args.lr,
                        'g_lr': args.g_lr,
                        'batch_size': args.batch_size,
                        'optim_type': args.optim_type,
                        'datasize': len(train_data),
                        'outdim': out_dim,
                        'seed': args.seed,
                        'model_save_dir': model_save_dir,
                        'exp_id': exp_id,
                        'save_dir': args.save_dir
    }

    #for mcmc techniques (per client)
    #num_mcmc_epochs = args.num_rounds * args.num_epochs_per_client

    mcmc_hyperparams = { 'epoch_per_client': args.epoch_per_client,
                    'weight_decay': 5e-4,
                    'datasize': len(train_data),
                    'batch_size': args.batch_size, #100
                    'init_lr': args.lr, #0.1, #0.5
                    'M': args.num_cycles, #5, #4, # num_cycles 
                    'sample_per_cycle': args.sample_per_cycle,
                    'temp':args.temp,
                    'alpha': 0.9,
                    'max_samples': args.max_samples,
                    'outdim': out_dim,
                    'seed': args.seed,
                    'exp_id': exp_id,
                    'model_save_dir': model_save_dir,
                    'save_dir' : args.save_dir,
                    'rho': args.rho, #below 3 are for fed_pa
                    'global_lr': args.g_lr,
                    'optim_type': args.optim_type,
                    'seed': args.seed,
                    'dataset': args.dataset,
                    'non_iid': args.non_iid
    }

    #do this for all datasets for fairness to the distillation algos
    #split train data into distill and train
    len_data = train_data.__len__()
    len_more_data = int(round(len_data*0.2))
    lens = [len_data - len_more_data, len_more_data]
    train_data, distill_data = torch.utils.data.random_split(train_data, lens)

    # setup a validation set for hyperparameter tuning
    VALIDATE = False

    if VALIDATE:
        #if not a distillation dataset, use distill data for validation
        if mode not in ["distill_f_mcmc", "oneshot_fl", "oneshot_fl_cs"]: 
            val_data_tuning = distill_data

            #may want to change pin_memory to false if not using cuda
            valloader_tuning = torch.utils.data.DataLoader(val_data_tuning, 
                                                            batch_size=args.batch_size, 
                                                            shuffle=False, pin_memory=True, 
                                                            num_workers=3)

        # if is a distillation algo, use half of distill_data for tuning (for overall 10% val data size)
        # for testing, we switch back to using all of distillation set
        else:
            len_d_data = distill_data.__len__()
            len_d_more_data = int(round(len_d_data*0.5))
            lens = [len_d_data - len_d_more_data, len_d_more_data]
            distill_data, val_data_tuning = torch.utils.data.random_split(distill_data, lens)

            valloader_tuning = torch.utils.data.DataLoader(val_data_tuning, 
                                                            batch_size=args.batch_size, 
                                                            shuffle=False, pin_memory=True, 
                                                            num_workers=3)

    ################################
    # TRAINING ALGORITHMS
    ################################
    if mode == "sgd":
        
        base_net = train_nets.sgd_train(base_net, args.lr, args.epoch_per_client*args.num_rounds, trainloader)
        
        acc = utils.classify_acc(base_net, valloader)
    elif mode == "fed_sgd":
        sgd_hyperparams['device'] = device

        fed_avg_trainer = fed_algos.FedAvg(num_clients = args.num_clients, 
                                        base_net = base_net, 
                                        traindata = train_data, 
                                        num_rounds = args.num_rounds, 
                                        hyperparams = sgd_hyperparams, 
                                        logger = logger,
                                        non_iid = args.non_iid,
                                        task = task)
        #fed_avg_trainer.train(valloader)
        fed_avg_trainer.train_saved_models(valloader)
        #acc = utils.classify_acc(fed_avg_trainer.global_net, valloader)
    elif mode == "fed_prox":
        sgd_hyperparams['device'] = device
        sgd_hyperparams['reg_global'] = args.prox_reg

        fed_avg_trainer = fed_algos.FedProx(num_clients = args.num_clients, 
                                        base_net = base_net, 
                                        traindata = train_data, 
                                        num_rounds = args.num_rounds, 
                                        hyperparams = sgd_hyperparams, 
                                        logger = logger,
                                        non_iid = args.non_iid,
                                        task = task)

        if VALIDATE:
            #fed_avg_trainer.train(valloader_tuning)
            fed_avg_trainer.train_saved_models(valloader_tuning)
        else:
            #fed_avg_trainer.train(valloader)
            fed_avg_trainer.train_saved_models(valloader)
        #acc = utils.classify_acc(fed_avg_trainer.global_net, valloader)
        
    elif mode == "adapt_fl":
        sgd_hyperparams['device'] = device
        sgd_hyperparams['tau'] = args.tau #10e-3

        fed_avg_trainer = fed_algos.AdaptiveFL(num_clients = args.num_clients, 
                                        base_net = base_net, 
                                        traindata = train_data, 
                                        num_rounds = args.num_rounds, 
                                        hyperparams = sgd_hyperparams, 
                                        logger = logger,
                                        non_iid = args.non_iid,
                                        task = task)
        if VALIDATE:
            #fed_avg_trainer.train(valloader_tuning)
            fed_avg_trainer.train_saved_models(valloader_tuning)
        else:
            #fed_avg_trainer.train(valloader)
            fed_avg_trainer.train_saved_models(valloader)

        #acc = utils.classify_acc(fed_avg_trainer.global_net, valloader)
    elif mode == "fed_be":
        #len_data = train_data.__len__()
        #len_more_data = int(round(len_data*0.2))
        #lens = [len_data - len_more_data, len_more_data]
        #train_data, distill_data = torch.utils.data.random_split(train_data, lens)
        sgd_hyperparams['kd_lr'] = args.kd_lr
        sgd_hyperparams['kd_optim_type'] = args.kd_optim_type
        sgd_hyperparams['kd_epochs'] = args.kd_epochs
        oneshot_fl = fed_algos.FedBE(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data, distill_data = distill_data,
                                    num_rounds = 1,
                                    hyperparams = sgd_hyperparams, device=device, logger = logger,
                                    args=args, non_iid = args.non_iid,
                                    task = task)
        #oneshot_fl.train(valloader=valloader)
        oneshot_fl.train_saved_models_no_distill(valloader)
    elif mode == "fed_pa":
        #add hyperparameter
        mcmc_hyperparams['rho'] = args.rho #1.0
        mcmc_hyperparams['global_lr'] = args.g_lr #1e-1 #1.0

        #create new: globa optim type
        mcmc_hyperparams['optim_type'] = args.optim_type

        if args.num_rounds > 1:
            #change number of cycles to 1, since we do multiple rounds
            mcmc_hyperparams['M'] = 1

        fed_pa = fed_algos.FedPA(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data,
                                    num_rounds = args.num_rounds, #4,#1,
                                    hyperparams = mcmc_hyperparams, device=device, logger = logger,
                                    non_iid = args.non_iid,
                                    task = task)
        #fed_pa.train(valloader=valloader)
        fed_pa.train_saved_models(valloader)
    elif mode == "ep_mcmc":
        ep_mcmc = fed_algos.EP_MCMC(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data,
                                    num_rounds = 1,
                                    hyperparams = mcmc_hyperparams, device=device, logger = logger,
                                    non_iid = args.non_iid,
                                    task = task)
        ep_mcmc.train(valloader=valloader)

    elif mode == "oneshot_fl":

        #len_data = train_data.__len__()
        #len_more_data = int(round(len_data*0.2))
        #lens = [len_data - len_more_data, len_more_data]
        #train_data, distill_data = torch.utils.data.random_split(train_data, lens)
        sgd_hyperparams['kd_lr'] = args.kd_lr
        sgd_hyperparams['kd_optim_type'] = args.kd_optim_type
        sgd_hyperparams['kd_epochs'] = args.kd_epochs
        oneshot_fl = fed_algos.ONESHOT_FL(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data, distill_data = distill_data,
                                    num_rounds = 1,
                                    hyperparams = sgd_hyperparams, device=device, logger = logger,
                                    args=args, non_iid = args.non_iid,
                                    task = task)
        #oneshot_fl.train(valloader=valloader)
        oneshot_fl.train_saved_models_no_distill(valloader)
    elif mode == "oneshot_fl_cs":

        sgd_hyperparams['kd_lr'] = args.kd_lr
        sgd_hyperparams['kd_optim_type'] = args.kd_optim_type
        sgd_hyperparams['kd_epochs'] = args.kd_epochs
        oneshot_fl_cs = fed_algos.ONESHOT_FL_CS(num_clients=args.num_clients,
                                          base_net=base_net,
                                          traindata=train_data, distill_data=distill_data,
                                          num_rounds=1,
                                          hyperparams=sgd_hyperparams, device=device, logger=logger,
                                          args=args, non_iid=args.non_iid,
                                          task=task)
        oneshot_fl_cs.train(valloader=valloader)
    elif mode == "f_mcmc":

        f_mcmc = fed_algos.F_MCMC(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data,
                                    num_rounds = 1,
                                    hyperparams = mcmc_hyperparams, device=device, logger = logger,
                                    non_iid = args.non_iid,
                                    task = task)
        f_mcmc.train(valloader=valloader)
      
    elif mode == "distill_f_mcmc":
        
        #split train data into distill and train
        #len_data = train_data.__len__()
        #len_more_data = int(round(len_data*0.2))
        #lens = [len_data - len_more_data, len_more_data]
        #train_data, distill_data = torch.utils.data.random_split(train_data, lens)

        #additional hyperparams
        mcmc_hyperparams['kd_lr'] = args.kd_lr
        mcmc_hyperparams['kd_optim_type'] = args.kd_optim_type
        mcmc_hyperparams['kd_epochs'] = args.kd_epochs

        f_mcmc = fed_algos.F_MCMC_distill(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data, distill_data = distill_data,
                                    num_rounds = 1,
                                    hyperparams = mcmc_hyperparams, device=device, logger = logger,
                                    non_iid = args.non_iid,
                                    task = task)
        f_mcmc.train(valloader=valloader)

    elif mode == "tune_distill_f_mcmc":
        #split train data into distill and train
        #len_data = train_data.__len__()
        #len_more_data = int(round(len_data*0.2))
        #lens = [len_data - len_more_data, len_more_data]
        #train_data, distill_data = torch.utils.data.random_split(train_data, lens)
        #inp, pred = next(iter(trainloader))
        #inp_dim = inp.shape

        #additional hyperparams
        mcmc_hyperparams['kd_lr'] = args.kd_lr
        mcmc_hyperparams['kd_optim_type'] = args.kd_optim_type
        mcmc_hyperparams['kd_epochs'] = args.kd_epochs

        mcmc_hyperparams['init_interp_param'] = 0.5
        mcmc_hyperparams['interp_param_lr'] = 1e-2

        f_mcmc = fed_algos.Calibrated_PredBayes_distill(num_clients = args.num_clients,
                                    base_net = base_net,
                                    traindata=train_data, distill_data = distill_data,
                                    num_rounds = 1,
                                    hyperparams = mcmc_hyperparams, device=device, logger = logger,
                                    non_iid = args.non_iid,
                                    task = task)
        f_mcmc.train(valloader=valloader)
    elif mode == "genbayes":
        sgd_hyperparams['kd_lr'] = args.kd_lr
        sgd_hyperparams['kd_optim_type'] = args.kd_optim_type
        sgd_hyperparams['kd_epochs'] = args.kd_epochs
        sgd_hyperparams['gen_lr'] = args.gen_lr

        EMBEDDING_DIM = inp_dim # The dimension of the embeddings
        FLOW_N = 9 # Number of affine coupling layers
        RNVP_TOPOLOGY = [200] # Size of the hidden layers in each coupling layer

        #normalizing flow generative model
        base_gen = nvp.LinearRNVP(input_dim=EMBEDDING_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N, batch_norm=True,
                      mask_type='odds', conditioning_size=10, use_permutation=True, single_function=True,
                      task = task)
        base_gen = base_gen.to(device)

        genbayes = fed_algos.GenBayes(num_clients = args.num_clients,
                                    base_net = base_net, base_gen=base_gen, 
                                    traindata=train_data, distill_data = distill_data,
                                    num_rounds = 1,
                                    hyperparams = sgd_hyperparams, device=device, logger = logger,
                                    args=args, non_iid = args.non_iid,
                                    task = task)
        genbayes.train(valloader=valloader)
    elif mode == "global_bayes":
        print("cSGHMC inference")
        
        trainer = train_nets.cSGHMC(base_net=base_net, trainloader=trainloader, device=device)
        trainer.train()
        acc = trainer.test_acc(valloader)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12)
    
    parser.add_argument('--dataset', type= str, default = "mnist")
    parser.add_argument('--non_iid', type = float, default = 0.0) # percent of non-iid #action="store_true") 

    parser.add_argument('--mode', type=str, default = "fed_sgd")

    parser.add_argument('--net_type', type=str, default="fc")

    parser.add_argument('--g_lr', type= float, default = 1e-1)
    parser.add_argument('--rho', type=float, default= 1.0)

    #dataset stuff
    parser.add_argument('--batch_size', type=int, default = 100)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default = 5e-4)
    parser.add_argument('--optim_type', type= str, default="sgdm")

    parser.add_argument('--gen_lr', type=float, default=1e-4)

    #for federated learning
    parser.add_argument('--num_rounds', type=int, default = 6)
    parser.add_argument('--epoch_per_client', type=int, default = 4)

    parser.add_argument('--num_clients', type = int, default = 5)


    parser.add_argument('--save_dir', type=str, default = "./results/")

    parser.add_argument('--sample_per_cycle', type=int, default=2)
    parser.add_argument('--temp', type=float, default = -1) #-1 => 1/datasize
    parser.add_argument('--num_cycles', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default = 6)
    parser.add_argument('--kd_optim_type', type =str, default = "adam")
    parser.add_argument('--kd_lr', type=float, default=1e-4)
    parser.add_argument('--kd_epochs', type=int, default = 50)

    #for later - setting up train/test split
    #parser.add_argument('--ntrain', type=int, default=500)
    #parser.add_argument('--train_percent', type=float, default=0.8)

    #for FedProx
    parser.add_argument('--prox_reg', type=float, default=1e-2)

    #for adaptive FL (fedYogi)
    parser.add_argument('--tau', type= float, default = 10e-3)

    args = parser.parse_args()
    
    main(args)
