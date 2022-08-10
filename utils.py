import os 
import torch
import numpy as np
import pickle

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    return

# compute accuracy for a classification task
def classify_acc(net, dataloader):
    # test eval
    total = 0
    correct = 0

    #eval mode
    net = net.eval()
 
    for x,y in dataloader:
        x = x.to(device)
        y= y.to(device)

        pred_logit = net(x)
        _, pred = torch.max(pred_logit, 1)    

        total += y.size(0)
        correct += (pred == y).sum().item()

    acc = 100*correct/total
    print("Accuracy on test set: ", acc)
    return acc

# compute MSE loss for a regression task
def regr_acc(net, dataloader):
    # test eval

    #eval mode
    net = net.eval()
    criterion = torch.nn.MSELoss()

    total = 0.0

    for x,y in dataloader:
        x = x.to(device)
        y= y.to(device)

        pred = net(x)
        
        loss = criterion(pred, y)
        total += loss.item()
    print("MSE on test set: ", total)
    return total

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def print_and_log(s, logger):
    print(s)
    logger.write(str(s) + '\n')


def save_dict(dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dict, fname)
    return

def load_dict(fname):
    if os.path.isfile(fname):
        dict = pickle.load(open(fname,'rb'))
    else:
        dict = {}
    return dict

def write_result_dict(result, seed, logger_file):
    #parse file name of logger
    fname_dict = os.path.splitext(logger_file)[0]
    dict = load_dict(fname_dict)
    dict['seed'] = result
    save_dict(dict, fname_dict)
    return