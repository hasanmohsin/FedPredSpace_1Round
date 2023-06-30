import os 
import torch
import numpy as np
import pickle
import tensorflow_probability as tfp

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
        pickle.dump(dict, f)
    return

def load_dict(fname):
    if os.path.isfile(fname):
        dict = pickle.load(open(fname,'rb'))
    else:
        dict = {}
    return dict

#type of result dict: if "" its accuracy
def write_result_dict(result, seed, logger_file, type="acc"):
    
    if type == "acc":
        pickle_tag = "ACC.pickle"
    elif type == "nllhd":
        pickle_tag = "NLLHD.pickle"
    elif type == "cal":
        pickle_tag = "CAL.pickle"
    else:
        pickle_tag = ".pickle"

    #parse file name of logger

    fname_dict = os.path.splitext(logger_file.name)[0]+pickle_tag
    dict = load_dict(fname_dict)
    print("fname for dict: ", fname_dict)
    print("dict [{}] = {}".format(seed, result))
    dict["{}".format(seed)] = result
    save_dict(dict, fname_dict)
    return

def write_result_dict_to_file(result, seed, file_name, type="acc"):
    if type == "acc":
        pickle_tag = "ACC.pickle"
    elif type == "nllhd":
        pickle_tag = "NLLHD.pickle"
    elif type == "cal":
        pickle_tag = "CAL.pickle"
    else:
        pickle_tag = ".pickle"
    
    #parse file name of logger
    fname_dict = file_name + pickle_tag
    dict = load_dict(fname_dict)
    print("fname for dict: ", fname_dict)
    print("dict [{}] = {}".format(seed, result))
    dict["{}".format(seed)] = result
    save_dict(dict, fname_dict)
    return

#given the id string for an experiment, and its mode, convert to the id string of the target mode
def change_exp_id(exp_id_src, source_mode, target_mode):
    exp_id_target = exp_id_src.replace(source_mode, target_mode)
    #comps = exp_id_src.split('_')

    #for i in range(len(comps)):
    #    if comps[i] == source_mode:
    #        comps[i] = target_mode

    #exp_id_target = "_".join(comps)
    return exp_id_target


def seq_split(dataset, lengths, generator = torch.default_generator):
    """
    Split a dataset into non-overlapping new datasets of given lengths., with the order they currently have
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    #just the integers [0, 1, ..., datasize-1]
    indices_sort, _ = torch.sort(torch.randperm(sum(lengths), generator=generator))
    indices = indices_sort.tolist()
    return [torch.utils.data.Subset(dataset, indices[offset - length : offset]) for offset, length in zip(torch._utils._accumulate(lengths), lengths)]



# for evaluation of model calibration

#calculate expected calibration error for classification
def test_ece(model, testloader, device, model_type = "ensemble"):
    acc = [] #1 if accurate, 0 if not accurate
    conf = []
    gt = []

    #first iterate thru all points in test loader, gathering each points 
    # accuracy and confidence
    for batch_idx, (x,y) in enumerate(testloader):
        x = x.to(device)
        y = y.to(device)

        if model_type == "ensemble":
            pred = model.predict(x)
        else:
            pred = model(x)
            pred = torch.functional.F.softmax(pred)

        confidence, pred_class = torch.max(pred, 1)    

        correct = (pred_class == y).sum().item()

        if batch_idx == 0:
            #acc = np.array(correct)
            logits = np.log(pred.detach().cpu().numpy())
            gt = y.detach().cpu().numpy()
        else:
            #acc = np.concatenate((acc, correct), 0)
            #print(confidence)
            logits = np.concatenate((logits, np.log(pred.detach().cpu().numpy())), 0)
            gt = np.concatenate((gt, y.detach().cpu().numpy()), 0)

    #acc = np.array(acc)
    #conf = np.array(conf)
    #gt = np.array(gt)
    print("GT shape: ", gt.shape)
    print("Logits shape: ", logits.shape)
    ece = tfp.stats.expected_calibration_error(num_bins=15, labels_true=gt.reshape(-1), logits=logits)
    return ece 

def test_classify_nllhd(model, testloader, device, model_type = "ensemble"):
    total_loss = 0.0
    criterion = torch.nn.NLLLoss()

    for batch_idx,(x,y) in  enumerate(testloader):
        x = x.to(device)
        y = y.to(device)
        
        if model_type == "ensemble":
            pred = model.predict(x)
        else:
            pred = model(x)
            pred = torch.functional.F.softmax(pred) # should output probability
        
        #pred = pred.reshape(y.shape)
        total_loss += criterion(torch.log(pred), y).item()

    print("NLLHD on test set: ", total_loss)
    return total_loss    

def test_regr_nllhd(model, testloader, device, model_type = "ensemble"):
    total_loss = 0.0
    criterion = torch.nn.GaussianNLLLoss()

    for batch_idx,(x,y) in  enumerate(testloader):
        x = x.to(device)
        y = y.to(device)
        
        if model_type == "ensemble":

            pred, pred_var = model.predict(x)
            #print("pred shape: ", pred.shape)
            #print("pred_var shape: ", pred_var.shape)
        else:
            pred, pred_var = model(x)
        #pred = pred.reshape(y.shape)
        total_loss += criterion(pred, y, pred_var).item()

    print("NLLHD on test set: ", total_loss)
    return total_loss    

#for regression, check if the models predictions with 95% confidence are actually 95%
def test_cal_95(model, testloader, device, model_type = "ensemble"):
    std_num = 1.96 #for 95% interval
    
    num_in_interval_total = 0
    num_pts = 0

    for batch_idx, (x, y) in enumerate(testloader):
        x = x.to(device)
        y = y.to(device)

        if model_type == "ensemble":
            mean, var = model.predict(x)
        else:
            mean, var = model(x)

        #reshape to be similar to y predictions
        mean = mean.reshape(y.shape)
        var = var.reshape(y.shape)

        interval_bot = mean - std_num*torch.sqrt(var)
        interval_top = mean + std_num*torch.sqrt(var)

        within_interval = (y >= interval_bot) & (y <= interval_top)
        num_in_interval = within_interval.sum()
        
        #print("pred_mean shape: ", mean.shape)
        #print("y.shape: ", y.shape)

        num_in_interval_total += num_in_interval.detach().cpu().numpy()
        num_pts += y.shape[0]
    #utils.print_and_log("", logger=logger)

    #print("Num_in_interval: {}, num_pts: {}".format(num_in_interval_total, num_pts))
    frac_in_interval = num_in_interval_total/num_pts 
    error = (frac_in_interval - 0.95)
    
    print("Error (frac_in_interval - 0.95): {}".format(error))

    return frac_in_interval, error

def test_calibration(model, testloader, task, device, model_type = "ensemble"):
    if task == "classify":
        nllhd = test_classify_nllhd(model, testloader, device, model_type)
        ece = test_ece(model, testloader, device, model_type)
        return nllhd, ece
    else:
        nllhd = test_regr_nllhd(model, testloader, device, model_type)
        frac_interval, error_cal = test_cal_95(model, testloader, device, model_type)
        return nllhd, frac_interval


#KL(p||q) for univariate Gaussians
# note: means and vars should be of shape [B, 1] 
def kl_div_gauss(p_mean, p_var, q_mean, q_var):
    p_std = torch.sqrt(p_var)
    q_std = torch.sqrt(q_var) 
    kl_val = torch.log(q_std) - torch.log(p_std) + (p_var + (p_mean - q_mean)**2)/(2 * q_var) - 0.5 

    return kl_val.mean() 