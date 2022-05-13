#!/bin/bash

printf "\nRunning F MCMC Experiment: \n"
python run_exp.py --mode f_mcmc --dataset $1 --epoch_per_client 25 --lr 5e-1 --num_round 1 --non_iid 1.0 --save_dir "./noniid_results/"

printf "\nRunning Fed SGDM Experiment: \n"
python run_exp.py --mode fed_sgd --dataset $1 --epoch_per_client 5 --lr 1e-2 --num_round 5 --optim_type "sgdm" --non_iid 1.0 --save_dir "./noniid_results/"

printf "\nRunning FedPA Experiment: \n"
python run_exp.py --mode fed_pa --dataset $1 --epoch_per_client 5 --lr 5e-1 --num_round 5 --non_iid 1.0 --save_dir "./noniid_results/"

printf "\nRunning EP MCMC Experiment \n"
python run_exp.py --mode ep_mcmc --dataset $1 --epoch_per_client 25 --lr 5e-1 --num_round 1 --non_iid 1.0 --save_dir "./noniid_results/"
