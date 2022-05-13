import os 
import torch
import numpy as np

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
