#!/bin/bash

./noniid_client_5_exps.sh mnist > noniid_mnist_log.txt
#./client_5_exp.sh cifar10 > cifar10_log.txt
#./client_5_exp.sh cifar100 > cifar100_log.txt
./noniid_client_5_exps.sh emnist > noniid_emnist_log.txt
./noniid_client_5_exps.sh f_mnist > noniid_f_mnist_log.txt
#./client_5_exps.sh airquality > airquality_log.txt
#./client_5_exps.sh bike > bike_log.txt
#./client_5_exps.sh forest_fire > forest_fire_log.txt
#./client_5_exps.sh winequality > winequality_log.txt
#./client_5_exps.sh real_estate > real_estate_log.txt

