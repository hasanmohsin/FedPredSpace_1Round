
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