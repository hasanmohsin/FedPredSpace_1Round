#!/bin/bash

mode=fed_sgd


for non_iid in 0.0 0.3 0.6 0.9  
do
	if [ $non_iid == 0.0 ]
	then
		lr_mnist=1e-1
		lr_emnist=1e-1
		lr_cifar=1e-2
	else
		lr_mnist=1e-2
		lr_emnist=1e-3
		lr_cifar=1e-3
	fi

	for seed in 11 12 13 #14 15 16 17 18 19 20
	do

		printf "\nRunning $mode MNIST Experiment $seed: \n"
		python run_exp.py --mode $mode --dataset mnist --epoch_per_client 5 --lr $lr_mnist --num_round 5 --optim_type "sgdm" --non_iid $non_iid --seed $seed --save_dir "./results_tuned_other/"


		printf "\nRunning $mode FMNIST Experiment, seed $seed: \n"
		python run_exp.py --mode $mode --dataset f_mnist --epoch_per_client 5 --lr $lr_mnist --rho 1.0 --num_round 5 --optim_type "sgdm" --non_iid $non_iid --seed $seed --save_dir "./results_tuned_other/"

		printf "\nRunning $mode EMNIST Experiment, seed $seed: \n"
		python run_exp.py --mode $mode --dataset emnist --epoch_per_client 5 --lr $lr_emnist --num_round 5 --optim_type "sgdm" --non_iid $non_iid --seed $seed --save_dir "./results_tuned_other/"

		printf "\nRunning $mode CIFAR10 Experiment, seed $seed: \n"
		python run_exp.py --mode $mode --dataset cifar10 --epoch_per_client 5 --lr $lr_cifar --num_round 10 --optim_type "sgdm" --net_type cnn --non_iid $non_iid --seed $seed --save_dir "./results_tuned_other/"


		printf "\nRunning $mode CIFAR100 Experiment: \n"
		python run_exp.py --mode $mode --dataset cifar100 --epoch_per_client 5 --lr $lr_cifar --num_round 10 --optim_type "sgdm" --net_type cnn --non_iid $non_iid --seed $seed --save_dir "./results_tuned_other/"

	done
done

