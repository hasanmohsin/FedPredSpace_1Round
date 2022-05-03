import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import matplotlib
matplotlib.use('Agg')

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def main(args):
    
    print(args)
    
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12)
    
    #dataset stuff
    parser.add_argument('--batch_size', type=int, default = 50)
    parser.add_argument('--train_percent', type=float, default=0.8)

    parser.add_argument('--lr', type=float, default=1e-1)

    parser.add_argument('--n_iters', type=int, default = 100)

    parser.add_argument('--save_dir', type=str, default = "./plots/")

    parser.add_argument('--ntrain', type=int, default=500)

    args = parser.parse_args()
    
    main(args)
