#!/bin/bash

mode=adapt_fl

for non_iid in 0.0 0.3 0.6 0.9  
do
	if [ $non_iid == 0.0 ]
	then
		lr_mnist=1e-2
		lr_emnist=1e-2
		lr_cifar=1e-1

		g_lr_mnist=1e-1
		g_lr_emnist=1e-1
		g_lr_cifar=1e-1

		optim_mnist="sgdm"
		optim_emnist="sgdm"
		optim_cifar="sgd"

		tau_mnist=1e-2
		tau_emnist=1e-2
		tau_cifar=1e-2
	else
		lr_mnist=1e-1
		lr_emnist=1e-1
		lr_cifar=1e-1

		g_lr_mnist=1e-1
		g_lr_emnist=1e-1
		g_lr_cifar=1e-1

		optim_mnist="sgd"
		optim_emnist="sgd"
		optim_cifar="sgd"

		tau_mnist=1e-2
		tau_emnist=1e-2
		tau_cifar=1e-2
	fi

	for seed in 11 12 13 #14 15 16 17 18 19 20
	do
		printf "\nRunning $mode MNIST Experiment $seed: \n" 
		python run_exp.py --mode $mode --dataset mnist --epoch_per_client 25 --lr $lr_mnist --num_round 1 --optim_type $optim_mnist --non_iid $non_iid --seed $seed --g_lr $g_lr_mnist --tau $tau_mnist --save_dir "./results_tuned_other/"

		printf "\nRunning $mode FMNIST Experiment, seed $seed: \n"
		python run_exp.py --mode $mode --dataset f_mnist --epoch_per_client 25 --lr $lr_emnist --num_round 1 --optim_type $optim_emnist --non_iid $non_iid --seed $seed --g_lr $g_lr_emnist --tau $tau_emnist --save_dir "./results_tuned_other/"


		printf "\nRunning $mode EMNIST Experiment, seed $seed: \n" 
		python run_exp.py --mode $mode --dataset emnist --epoch_per_client 25 --lr $lr_emnist --num_round 1 --optim_type $optim_emnist --non_iid $non_iid --seed $seed --g_lr $g_lr_emnist --tau $tau_emnist --save_dir "./results_tuned_other/"


		printf "\nRunning $mode CIFAR10 Experiment, seed $seed: \n"
		python run_exp.py --mode $mode --dataset cifar10 --epoch_per_client 50 --lr $lr_cifar --num_round 1 --optim_type $optim_cifar --net_type cnn --non_iid $non_iid --seed $seed --g_lr $g_lr_cifar --tau $tau_cifar --save_dir "./results_tuned_other/"


		printf "\nRunning $mode CIFAR100 Experiment: \n"
		python run_exp.py --mode $mode --dataset cifar100 --epoch_per_client 50 --lr $lr_cifar --num_round 1 --optim_type $optim_cifar --net_type cnn --non_iid $non_iid --seed $seed --g_lr $g_lr_cifar --tau $tau_cifar --save_dir "./results_tuned_other/"


	done
done


