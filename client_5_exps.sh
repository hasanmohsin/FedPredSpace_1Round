#!/bin/bash

printf "\nRunning F MCMC Experiment: \n"
python run_exp.py --mode f_mcmc --dataset $1 --epoch_per_client 50 --lr 5e-1 --num_round 1 --net_type cnn

#printf "\nRunning Fed SGD (no momentum) Experiment: \n"
#python run_exp.py --mode fed_sgd --dataset $1 --epoch_per_client 10 --lr 1e-2 --num_round 5 --optim_type "sgd" --net_type cnn

printf "\nRunning Fed SGDM Experiment: \n"
python run_exp.py --mode fed_sgd --dataset $1 --epoch_per_client 10 --lr 1e-2 --num_round 5 --optim_type "sgdm" --net_type cnn

printf "\nRunning Fed SGDM Experiment 1 round: \n"
python run_exp.py --mode fed_sgd --dataset $1 --epoch_per_client 50 --lr 1e-2 --num_round 1 --optim_type "sgdm" --net_type cnn

printf "\nRunning FedPA Experiment: \n"
python run_exp.py --mode fed_pa --dataset $1 --epoch_per_client 10 --lr 5e-1 --num_round 5 --net_type cnn

printf "\nRunning FedPA Experiment 1 round: \n"
python run_exp.py --mode fed_pa --dataset $1 --epoch_per_client 50 --lr 5e-1 --num_round 1 --net_type cnn

printf "\nRunning EP MCMC Experiment \n"
python run_exp.py --mode ep_mcmc --dataset $1 --epoch_per_client 50 --lr 5e-1 --num_round 1 --net_type cnn
