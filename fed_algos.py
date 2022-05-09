import copy
import torch
import numpy as np
import datasets
import train_nets
import utils

class FedAvg:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, epoch_per_client,
                batch_size):

        lr = 0.01

        self.all_data = traindata

        self.num_clients = num_clients

        #initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []
        self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            
            self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = lr, momentum=0.9))


        self.num_rounds = num_rounds
        self.epoch_per_client = epoch_per_client

        self.criterion = torch.nn.CrossEntropyLoss()
        self.global_net = copy.deepcopy(base_net)

    #perform 1 epoch of updates on client_num
    def local_update_step(self, client_num):
        
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net = self.client_nets[client_num],
                                 optimizer=self.optimizers[client_num],
                                 criterion = self.criterion,
                                 trainloader=c_dataloader)

        print("Client {}, Loss: {}".format(client_num, loss))

        return

    #in aggregation step - average all models
    def alt_aggregate(self):
        global_state_dict = self.global_net.state_dict()

        for layer in global_state_dict:
            global_state_dict[layer] = 0*global_state_dict[layer]

            #average over clients    
            for c in range(self.num_clients):
                global_state_dict[layer] += self.client_nets[c].state_dict()[layer]/self.num_clients

        self.global_net.load_state_dict(global_state_dict)

        return

    def aggregate(self):
        #in aggregation step - average all models
        #global_v = 0.0 #torch.nn.utils.paramters_to_vector(self.global_net.parameters())

        c_vectors = []

        #average over clients    
        for c in range(self.num_clients):
                c_vector = torch.nn.utils.parameters_to_vector(self.client_nets[c].parameters()).detach()
                c_vectors.append(torch.clone(c_vector))
        c_vectors = torch.stack(c_vectors, dim=0)
        global_v = torch.mean(c_vectors, dim=0)
        
        #load into global net
        torch.nn.utils.vector_to_parameters(global_v, self.global_net.parameters())

        return

    def global_update_step(self):
        local_infos = []
        
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_update_step(client_num)

        self.aggregate()
        
        #self.aggregate()
    
    def global_to_clients(self):
        for c in range(self.num_clients):
            self.client_nets[c].load_state_dict(self.global_net.state_dict())

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            self.global_to_clients()
            acc = utils.classify_acc(self.global_net, valloader)
            print("Global rounds completed: {}, test_acc: {}".format(i, acc))
        return

class EP_MCMC:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, epoch_per_client,
                batch_size, device):
        
        self.all_data = traindata

        self.num_clients = num_clients

        #initialize nets and data for all clients
        self.client_train = []
        self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, batch_size)

        for c in range(num_clients):
            self.client_train.append(train_nets.cSGHMC(copy.deepcopy(base_net), 
                                                        trainloader=self.client_dataloaders[c],
                                                        device = device))

        self.num_rounds = num_rounds
        self.epoch_per_client = epoch_per_client

        self.criterion = torch.nn.CrossEntropyLoss()
        self.global_train = train_nets.cSGHMC(copy.deepcopy(base_net), 
                                            trainloader =None,
                                            device = device)#copy.deepcopy(base_net)
        self.base_net = base_net
        self.num_g_samples = 10

    def local_train(self, client_num):
        self.client_train[client_num].train()

    def get_client_sample_mean_cov(self, client_num):
        #a list of sampled nets from training client
        client_samples = self.client_train[client_num].sampled_nets
        c_vectors = []
        for sample in client_samples:
            sample_net = copy.deepcopy(self.base_net)
            sample_net.load_state_dict(sample)
            c_vec = torch.nn.utils.parameters_to_vector(sample_net.parameters())
            c_vectors.append(c_vec)
        c_vectors = torch.stack(c_vectors, dim=0)
        mean = torch.mean(c_vectors, dim=0)
        
        #too memory intensive - approximate with diagonal matrix
        #cov = torch.Tensor(np.cov((c_vectors).detach().numpy().T))

        cov = torch.var(c_vectors, dim = 0)#.diag()
        return mean, cov

    def aggregate(self):
        #in aggregation step - average all models
        #global_v = 0.0 #torch.nn.utils.paramters_to_vector(self.global_net.parameters())

        global_prec = 0.0
        global_mean = 0.0

        #average over clients    
        for c in range(self.num_clients):
            mean, cov = self.get_client_sample_mean_cov(c)
            client_prec = 1/cov #torch.inv(cov)
            global_prec += client_prec
            global_mean += client_prec * mean #client_prec@mean
        
        global_mean = (1/global_prec) * global_mean #torch.inv(global_prec) @ global_mean
        global_var = (1/global_prec)

        dist = torch.distributions.Normal(global_mean, global_var.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)
        #dist = torch.distributions.MultivariateNormal(loc = global_mean, precision_matrix=global_prec)
        global_samples = dist.sample([self.num_g_samples])
        
        #print("global mean shape: ", global_mean.shape)
        #print("global var shape: ", global_var.shape)
        #print("global samples shape: ", global_samples.shape)
    
        self.global_train.sampled_nets = []
        
        for s in range(self.num_g_samples):
            sample = global_samples[s,0,:]

            #load into global net
            torch.nn.utils.vector_to_parameters(sample, self.global_train.net.parameters())
            self.global_train.sampled_nets.append(copy.deepcopy(self.global_train.net.state_dict()))

        return

    def global_update_step(self):
        for client_num in range(self.num_clients):
            self.local_train(client_num)

        self.aggregate()
    
    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            acc = self.global_train.test_acc(valloader)
            print("Global rounds completed: {}, test_acc: {}".format(i, acc))

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                print("Client {}, test accuracy: {}".format(c, acc_c))
        return

#ours
# same as EP_MCMC, but inference step is different
class F_MCMC(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, epoch_per_client,
                batch_size, device):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, epoch_per_client,
                batch_size, device)

    #do nothing in aggregate function
    def aggregate(self):
        return
    
    #prediction on input x
    def predict(self, x):
        global_pred = 1.0
        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred *= pred
        return global_pred/torch.sum(global_pred)

    def test_acc(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)    

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            acc = self.test_acc(valloader)
            print("Global rounds completed: {}, test_acc: {}".format(i, acc))

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                print("Client {}, test accuracy: {}".format(c, acc_c))
        return


#class FedPA(FedAvg):

#class PVI(FedAvg):
