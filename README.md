# FedFuncSpace_1Round

Run code using the command

```
python run_exp.py --mode f_mcmc --dataset <DATASET> --epoch_per_client 25 --lr 2e-1 --num_round 1 --seed 12345
```

- f_mcmc for our method
- ep_mcmc for Embarrasingly Parallel MCMC
- fed_sgd for FedAvg
- fed_pa for Federated Posterior Averaging
