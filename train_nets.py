import torch
import numpy as np
import copy

def sgd_train_step(net, optimizer, criterion, trainloader):
    epoch_loss = 0.0
    count = 0
    for x, y in trainloader:
        
        optimizer.zero_grad()
        pred_logits = net(x)
        loss = criterion(pred_logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        count+=1
    return net, epoch_loss
   

# training with regular SGD
def sgd_train(net, lr, num_epochs, trainloader):
    #optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr =lr)
    net = net.train()

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(num_epochs):
        net, epoch_loss = sgd_train_step(net, optimizer, criterion, trainloader)
        print("Epoch: ", i+1, "Loss: ", epoch_loss)
    
    print("Training Done!")
    return net


#cSGHMC
#code adapted from https://github.com/ruqizhang/csgmcmc/blob/master/experiments/cifar_csghmc.py

class cSGHMC:
    def __init__(self,
                  base_net, 
                  trainloader, device):
        self.net = base_net
        self.trainloader = trainloader
        self.device = device
    
        #hard-coded params
        self.num_epochs = 24
        self.weight_decay = 5e-4
        self.datasize = 60000
        self.batch_size = 100
        self.num_batch = self.datasize/self.batch_size + 1
        self.init_lr = 0.5
        self.M = 4 #num_cycles
        self.cycle_len = (self.num_epochs/self.M)
        self.T = self.num_epochs*self.num_batch
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temperature = 1/self.datasize
        self.alpha = 0.9

        self.max_samples = 15
        self.sampled_nets = []

    #gradient rule for SG Hamiltonian Monte Carlo
    def update_params(self, lr,epoch):
        for p in self.net.parameters():
            if not hasattr(p,'buf'):
                p.buf = torch.zeros(p.size()).to(self.device)
            d_p = p.grad.data
            d_p.add_(self.weight_decay, p.data)
            buf_new = (1-self.alpha)*p.buf - lr*d_p
            if (epoch%self.cycle_len)+1>4:
                eps = torch.randn(p.size()).to(self.device)
                buf_new += (2.0*lr*self.alpha*self.temperature/self.datasize)**.5*eps
            p.data.add_(buf_new)
            p.buf = buf_new

    #learning rate schedule according to cyclic SGMCMC
    def adjust_learning_rate(self, epoch, batch_idx):
        rcounter = epoch*self.batch_size+batch_idx
        cos_inner = np.pi * (rcounter % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5*cos_out*self.init_lr
        return lr

    def train_epoch(self, epoch):
        #print('\nEpoch: %d' % epoch)
        self.net = self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.net.zero_grad()
            lr = self.adjust_learning_rate(epoch,batch_idx)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.update_params(lr,epoch)

            train_loss += loss.data.item()
            #_, predicted = torch.max(outputs.data, 1)
            #total += targets.size(0)
            #correct += predicted.eq(targets.data).cpu().sum()
            #if batch_idx%100==0:
            #    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #        % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))
        print("Epoch: {}, Loss: {}".format(epoch+1, train_loss))
        #self.test_acc(self.trainloader)
    
    
    #code adapted from https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics/model.py
    def sample_net(self):

        if len(self.sampled_nets) >= self.max_samples:
            self.sampled_nets.pop(0)

        self.sampled_nets.append(copy.deepcopy(self.net.state_dict()))

        print("Sampled net, total num sampled: {}".format(len(self.sampled_nets)))

        return None

    #for each net in ensemble, compute prediction (could be logits depending on net)
    def ensemble_inf(self, x, Nsamples=0, out_probs = True):
        if Nsamples == 0:
            Nsamples = len(self.sampled_nets)

        x = x.to(self.device)

        out = x.data.new(Nsamples, x.shape[0], self.net.out_dim)

        # iterate over all saved weight configuration samples
        for idx, weight_dict in enumerate(self.sampled_nets):
            if idx == Nsamples:
                break
            self.net.load_state_dict(weight_dict)
            self.net.eval()
            out[idx] = self.net(x)

            if out_probs:
                out[idx] = torch.nn.functional.softmax(out[idx], dim = 1)


        return out

    def test_acc(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            pred_list = self.ensemble_inf(x, out_probs = True)

            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            _, pred_class = torch.max(pred, 1)    



            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc

    def train(self):

        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)

            # 3 models sampled every 8 epochs
            if (epoch%self.cycle_len)+1 > 4:
                self.sample_net()
        
        return 
#####################################################

def sgld_train(net, lr, num_epochs, trainloader):
    ###################
    # params
    alpha = 5e-2
    beta = 1e-1
    max_nsamples = 50
    burnin_epochs = 5
    sample_gap = 2 # sample model every 2 epochs

    ###################

    #collect sample nets
    sample_nets = []
    nsamples = 0

    optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    
    net = net.train()

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for x, y in trainloader:
            
            optimizer.zero_grad()
            pred_logits = net(x)
            loss = criterion(pred_logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            count+=1
        if i % sample_gap == 0 and i > burnin_epochs:
            if len(sample_nets) < max_nsamples:
                sample_nets.append()
                nsamples += 1 
            else:
                sample_nets[nsamples]
        
        
        print("Epoch: ", i+1, "Loss: ", epoch_loss)
    print("Training Done!")

    return nets

#######################################
# Fed AVG
#######################################

