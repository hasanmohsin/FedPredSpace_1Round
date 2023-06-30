import copy
import torch
import numpy as np
import datasets
import train_nets
import utils
import kd 
import nvp
import itertools
import models

class FedAvg:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, logger, non_iid = 0.0, task = "classify"):

        # lr for SGD
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.datasize = hyperparams['datasize']
        self.optim_type = hyperparams['optim_type']
        self.outdim = hyperparams['outdim']
        self.device = hyperparams['device']
        self.seed = hyperparams['seed']
        self.model_save_dir = hyperparams['model_save_dir']
        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        self.logger = logger

        self.all_data = traindata

        self.num_clients = num_clients

        self.task = task

        #initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            
            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                 self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr = self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))

        self.num_rounds = num_rounds

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    #perform 1 epoch of updates on client_num
    def local_update_step(self, client_num):
        
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net = self.client_nets[client_num],
                                 optimizer=self.optimizers[client_num],
                                 criterion = self.criterion,
                                 trainloader=c_dataloader, device = self.device)

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

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            #last round, save models before sending to clients
            if i == self.num_rounds - 1:
                self.save_models()

            self.global_to_clients()
            acc = self.get_acc(self.global_net, valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i+1, acc), self.logger)
        
        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)
        #utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        
        #self.save_models()
        
        return


class FedProx:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, logger, non_iid = 0.0, task = "classify"):

        # lr for SGD
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.datasize = hyperparams['datasize']
        self.optim_type = hyperparams['optim_type']
        self.outdim = hyperparams['outdim']
        self.device = hyperparams['device']
        self.seed = hyperparams['seed']
        self.model_save_dir = hyperparams['model_save_dir']
        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        self.reg_global = hyperparams['reg_global']

        self.logger = logger

        self.all_data = traindata

        self.num_clients = num_clients

        self.task = task

        #initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            
            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                 self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr = self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))

        self.num_rounds = num_rounds

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        #initial global net is same as client nets
        self.global_net = copy.deepcopy(base_net)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    #perform 1 epoch of updates on client_num
    def local_update_step(self, client_num):
        
        c_dataloader = self.client_dataloaders[client_num]

        #take optimization step regularized by global net
        self.client_nets[client_num], loss = train_nets.sgd_prox_train_step(net = self.client_nets[client_num], 
                                g_net = self.global_net,
                                reg = self.reg_global,
                                optimizer=self.optimizers[client_num],
                                criterion = self.criterion,
                                trainloader=c_dataloader, device = self.device)

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

    #updates global net
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

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            #last round, save models before sending to clients
            if i == self.num_rounds - 1:
                self.save_models()

            self.global_to_clients()
            acc = self.get_acc(self.global_net, valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i+1, acc), self.logger)
        
        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)
        #utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        
        #self.save_models()
        
        return

class AdaptiveFL:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, logger, non_iid = 0.0, task = "classify"):

        # lr for SGD
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.datasize = hyperparams['datasize']
        self.optim_type = hyperparams['optim_type']
        self.outdim = hyperparams['outdim']
        self.device = hyperparams['device']
        self.seed = hyperparams['seed']
        self.model_save_dir = hyperparams['model_save_dir']
        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        self.logger = logger

        self.all_data = traindata

        self.num_clients = num_clients

        self.task = task

        #initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            
            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                 self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr = self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr = self.lr))

        self.num_rounds = num_rounds

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.g_vec = torch.nn.utils.parameters_to_vector(self.global_net.parameters()).detach()
        
        # parameters for server optimization
        
        self.g_v = torch.zeros_like(self.g_vec)
        self.g_momentum = torch.zeros_like(self.g_vec)

        self.tau = hyperparams['tau'] #10e-3
        self.g_beta1 = 0.9
        self.g_beta2 = 0.99

        self.global_optimizer = torch.optim.Adam(self.global_net.parameters(), lr = self.g_lr)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    #perform 1 epoch of updates on client_num
    def local_update_step(self, client_num):
        
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net = self.client_nets[client_num],
                                 optimizer=self.optimizers[client_num],
                                 criterion = self.criterion,
                                 trainloader=c_dataloader, device = self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

        return
    
    #set delta as the gradient in the global optimizer
    def global_opt(self, g_delta):
        self.global_optimizer.zero_grad()

        #set global optimizer grad
        torch.nn.utils.vector_to_parameters(g_delta, self.delta.parameters())

        #copy gradient data over to global net
        for p, g in zip(self.global_net.parameters(), self.delta.parameters()):
            #print("p parmaeter ", p)
            #print("p grad paramter", p.grad)
            #print("p grad param.data ", p.grad.data)
            p.grad = g.data

        #update
        self.global_optimizer.step()

        return 
    
    def fed_yogi_update(self, g_delta):
        self.g_momentum = self.g_beta1 * self.g_momentum + (1-self.g_beta1) * g_delta
        self.g_v = self.g_v - (1 - self.g_beta2) * (g_delta**2) * torch.sign( self.g_v - g_delta**2 )       
    
        self.g_vec = torch.nn.utils.parameters_to_vector(self.global_net.parameters()).detach()
        self.g_vec = self.g_vec + self.g_lr * self.g_momentum/( torch.sqrt(self.g_v) + self.tau )
        
        #save to global net
        torch.nn.utils.vector_to_parameters(self.g_vec, self.global_net.parameters())

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
        global_vec = torch.nn.utils.parameters_to_vector(self.global_net.parameters()).detach()
        delta = torch.mean(c_vectors, dim=0) - global_vec
        
        self.fed_yogi_update(g_delta = delta)

        return

    def global_update_step(self):
        local_infos = []
        
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_update_step(client_num)

        self.aggregate()
        
    
    def global_to_clients(self):
        for c in range(self.num_clients):
            self.client_nets[c].load_state_dict(self.global_net.state_dict())

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            #last round, save models before sending to clients
            if i == self.num_rounds - 1:
                self.save_models()

            self.global_to_clients()
            acc = self.get_acc(self.global_net, valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i+1, acc), self.logger)
        
        #save final results here
        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)

        #self.save_models()
        
        return

class EP_MCMC:
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, 
                hyperparams, device, logger, non_iid = 0.0, task = "classify"):
        self.logger = logger
        self.all_data = traindata

        self.seed = hyperparams['seed']
        self.dataset = hyperparams['dataset']
        self.non_iid = hyperparams['non_iid']

        self.device = device

        self.num_clients = num_clients

        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.seed = hyperparams['seed']
        self.exp_id = hyperparams['exp_id']
    
        self.model_save_dir = hyperparams['model_save_dir']
        self.save_dir = hyperparams['save_dir']

        #initialize nets and data for all clients
        self.client_train = []
        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset = traindata, 
                                                                                        num_clients = num_clients, 
                                                                                        client_data_size = (self.datasize//num_clients), 
                                                                                        batch_size = self.batch_size, 
                                                                                        shuffle=False, non_iid_frac = non_iid,
                                                                                        outdim = self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        self.max_samples = hyperparams['max_samples']

        for c in range(num_clients):
            hyperparams_c = copy.deepcopy(hyperparams)
            #hyperparams_c['datasize'] = self.client_datasize[c]
            self.client_train.append(train_nets.cSGHMC(copy.deepcopy(base_net), 
                                                        trainloader=self.client_dataloaders[c],
                                                        device = device, task = task, hyperparams=hyperparams_c))

        self.num_rounds = num_rounds

        self.task = task

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()
        
        self.global_train = train_nets.cSGHMC(copy.deepcopy(base_net), 
                                            trainloader =None,
                                            device = device, task = task, hyperparams=hyperparams)#copy.deepcopy(base_net)
        self.base_net = base_net
        self.num_g_samples = 10


    def save_models(self):
        save_dir = self.model_save_dir
        save_name = utils.change_exp_id(self.exp_id, "distill_f_mcmc", "mcmc") + "_seed_" + str(self.seed)  
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            cpath = c_save + "_client_" + str(c)  

            for idx, weight_dict in enumerate(self.client_train[c].sampled_nets):
                path = cpath + "_sample_" + str(idx) 
                torch.save(weight_dict, path)


    def local_train(self, client_num):
        #trains for above specified epochs per client 
        self.client_train[client_num].train()

    def get_client_samples_as_vec(self, client_num):
        client_samples = self.client_train[client_num].sampled_nets
        c_vectors = []
        for sample in client_samples:
            sample_net = copy.deepcopy(self.base_net)
            sample_net.load_state_dict(sample)
            c_vec = torch.nn.utils.parameters_to_vector(sample_net.parameters())
            c_vectors.append(c_vec)
        c_vectors = torch.stack(c_vectors, dim=0)

        return c_vectors

    def get_client_sample_mean_cov(self, client_num):
        #a list of sampled nets from training client
        c_vectors = self.get_client_samples_as_vec(client_num)
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
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return

#ours
# same as EP_MCMC, but inference step is different
class F_MCMC(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, 
                device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, device, logger, non_iid, task)

    #do nothing in aggregate function
    def aggregate(self):
        return
    
    #prediction on input x
    def predict_classify(self, x):
        global_pred = 1.0
        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)

            #assuming a uniform posterior
            global_pred *= pred
        return global_pred/torch.sum(global_pred, dim=-1, keepdims=True)

    def predict_regr(self,x):
        global_pred = 0.0
        var_sum = 0.0

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred_mean = torch.mean(pred_list, dim=0, keepdims=False)
            pred_var = torch.var(pred_list, dim = 0, keepdims = False)

            #assuming a uniform posterior
            global_pred += pred_mean/pred_var
            var_sum += 1/pred_var
        
        return global_pred/var_sum


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)
            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc
    
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx,(x,y) in  enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.predict(x)
           
            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss    

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)


    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            acc = self.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return


#ours
# same as EP_MCMC, but inference step is different
class F_MCMC_distill(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, distill_data,
                num_rounds, hyperparams, 
                device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, device, logger, non_iid, task)

        #make distillation dataset out of distill_data
        distill_loader = torch.utils.data.DataLoader(distill_data, 
                                                batch_size=self.batch_size, shuffle=True, 
                                                pin_memory=True)

        self.student = copy.copy(base_net)
        self.kd_optim_type = hyperparams['kd_optim_type']
        self.kd_lr = hyperparams['kd_lr']
        self.kd_epochs = hyperparams['kd_epochs']

        self.distill = kd.KD(teacher = self, 
                             student = self.student, lr = self.kd_lr,#5e-3
                             device = self.device,
                             train_loader = distill_loader,
                             kd_optim_type = self.kd_optim_type
                            )

        self.fed_pa_trainer = FedPA(num_clients = num_clients, base_net = base_net, 
                                    traindata = traindata, 
                                    num_rounds = 1, 
                                    hyperparams = hyperparams, logger=None, non_iid = non_iid, task = self.task,
                                    device= self.device)

    #do nothing in aggregate function
    def aggregate(self):
        #try better student init
        self.distill.set_student(self.client_train[0].sampled_nets[-1])

        #train the student via kd
        self.distill.train(num_epochs = self.kd_epochs) #kd_epochs = 50
        self.student = self.distill.student

        return
    
    #so we can compare with EP MCMC 
    def ep_mcmc_aggregate(self):
        #in aggregation step - average all models
        #global_v = 0.0 #torch.nn.utils.paramters_to_vector(self.global_net.parameters())

        global_prec = 0.0
        global_mean = 0.0

        #average over clients    
        for c in range(self.num_clients):
            mean, cov = self.get_client_sample_mean_cov(c)
            client_prec = 1 #/self.num_clients #1/cov #torch.inv(cov) ########################################
            global_prec += client_prec
            global_mean += client_prec * mean #client_prec@mean
            #print("Client {} mean {}".format(c, mean))

        global_mean = (1/global_prec) * global_mean #torch.inv(global_prec) @ global_mean
        global_var = (1/global_prec)*torch.ones_like(global_mean).to(self.device)

        dist = torch.distributions.Normal(global_mean, global_var.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)
        #dist = torch.distributions.MultivariateNormal(loc = global_mean, precision_matrix=global_prec)
        global_samples = dist.sample([self.num_g_samples])
        
        #print("global mean shape: ", global_mean.shape)
        #print("global var shape: ", global_var.shape)
        #print("global samples shape: ", global_samples.shape)
    
        self.global_train.sampled_nets = []
        
        for s in range(self.num_g_samples):
            sample = global_mean #global_samples[s,0,:] #######################################

            #load into global net
            torch.nn.utils.vector_to_parameters(sample, self.global_train.net.parameters())
            self.global_train.sampled_nets.append(copy.deepcopy(self.global_train.net.state_dict()))

        return

    #prediction on input x
    def predict_classify(self, x):
        global_pred = 1.0
        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred *= pred
        return global_pred/torch.sum(global_pred, dim=-1, keepdims=True)


    def predict_regr(self,x):
        global_pred = 0.0
        var_sum = 0.0

        #pred_list_prior = self.client_train[c].ensemble_inf(x,)

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
            
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred_mean = torch.mean(pred_list, dim=0, keepdims=False)

            #the variance in the prediction is 1.0 (the homeoscedatic assumed variance) + bayesian variance
            pred_var = 1.0 + torch.var(pred_list, dim = 0, keepdims = False)

            #assuming a uniform posterior
            global_pred += pred_mean/pred_var
            var_sum += 1/pred_var

        return global_pred/var_sum


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)
            _, pred_class = torch.max(pred, 1)    

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc
    
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx,(x,y) in  enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.predict(x)
           
            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss    

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)
        

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            acc = self.distill.test_acc(valloader)
            #acc = self.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, distilled_f_mcmc test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)

        #evaluate samples on other methods (FMCMC, and EP MCMC)
        f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, f_mcmc test_acc: {}".format(i, f_mcmc_acc), self.logger)
        
        #save to dict
        utils.write_result_dict_to_file(result=f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"))

        #compute ep mcmc result and store in global_train
        self.ep_mcmc_aggregate()
        ep_mcmc_acc = self.global_train.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, ep_mcmc test_acc: {}".format(i, ep_mcmc_acc), self.logger)
        
        #save to dict 
        utils.write_result_dict_to_file(result=ep_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"))


        #for fed_pa 1 round results
        #copy over trained clients
        self.fed_pa_trainer.client_train = copy.deepcopy(self.client_train)

        #take step of FedPA 
        self.fed_pa_trainer.global_update_step_trained_clients()
        fed_pa_acc = self.fed_pa_trainer.get_acc(self.fed_pa_trainer.global_train.net, valloader)
        utils.print_and_log("Global rounds completed: {}, fed_pa test_acc: {}".format(i, fed_pa_acc), self.logger)
        
        #save to dict 
        utils.write_result_dict_to_file(result=fed_pa_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"))


        #save client sample models 
        self.save_models()

        return



#ours
# same as EP_MCMC, but inference step is different
class Gen_F_MCMC_distill(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, distill_data,
                inp_dim,
                num_rounds, hyperparams, 
                device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, device, logger, non_iid, task)

        #make distillation dataset out of distill_data
        distill_loader = torch.utils.data.DataLoader(distill_data, 
                                                batch_size=self.batch_size, shuffle=True, 
                                                pin_memory=True)

        self.student = copy.copy(base_net)
        self.distill = kd.KD_gen_data(teacher = self, 
                             student = self.student, lr = 5e-3,
                             device = self.device,
                             train_loader = distill_loader,
                             inp_dim = inp_dim
                            )

    #do nothing in aggregate function
    def aggregate(self):
        #try better student init
        self.distill.set_student(self.client_train[0].sampled_nets[-1])

        #train the student via kd
        self.distill.train(num_epochs = 50)
        self.student = self.distill.student

        return
    
    #prediction on input x
    def predict_classify(self, x):
        global_pred = 1.0
        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred *= pred
        return global_pred/torch.sum(global_pred, dim=-1, keepdims=True)


    def predict_regr(self,x):
        global_pred = 0.0
        var_sum = 0.0

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred_mean = torch.mean(pred_list, dim=0, keepdims=False)
            pred_var = torch.var(pred_list, dim = 0, keepdims = False)

            #assuming a uniform posterior
            global_pred += pred_mean/pred_var
            var_sum += 1/pred_var
        
        return global_pred/var_sum


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)    

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc
    
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx,(x,y) in  enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred = self.predict(x)
           
            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss    

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)


    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()

            acc = self.distill.test_acc(valloader)
            #acc = self.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return

#Federated posterior averaging
class FedPA(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, 
                hyperparams, device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams,
                 device, logger, non_iid, task)

        self.rho = hyperparams['rho'] # 1.0
        self.global_lr = hyperparams['global_lr'] # global learning rate should likely be higher
        self.global_train.net.requires_grad = True

        self.g_optim_type = hyperparams['optim_type']

        if self.g_optim_type == "sgdm":
            self.global_optimizer = torch.optim.SGD(self.global_train.net.parameters(), lr = self.global_lr, momentum=0.9)
        elif self.g_optim_type == "sgd":
            self.global_optimizer = torch.optim.SGD(self.global_train.net.parameters(), lr = self.global_lr)
        elif self.g_optim_type == "adam":
            self.global_optimizer = torch.optim.Adam(self.global_train.net.parameters(), lr = self.global_lr)
        
        #self.global_optimizer = torch.optim.SGD(params=self.global_train.net.parameters(), lr = self.global_lr, momentum=0.9)
        self.seed = hyperparams['seed']

    def get_global_vec(self):
        #sample_net = copy.deepcopy(self.base_net)
        #sample_net.load_state_dict(sample)
        g_vec = torch.nn.utils.parameters_to_vector(self.global_train.net.parameters())
        return g_vec

    #compute local delta for client_num, based on its samples
    def local_delta(self, client_num):
        #client samples as tensor N x num_params
        c_vectors = self.get_client_samples_as_vec(client_num)
        global_vec = self.get_global_vec()

        #first compute sample means, 
        #ranges = torch.arange(1, c_vectors.shape[0]+1).reshape(-1,1)
        #sample_means = torch.cumsum(c_vectors, dim=  0)/ranges
        
        num_samples = c_vectors.shape[0]
        
        #initialize
        delta_sim = global_vec - c_vectors[0,:]
        rhot = 1
        u = c_vectors[0,:]
        u_vecs = torch.clone(u).reshape(1, -1)

        if num_samples == 1:
            return delta_sim/rhot

        v = c_vectors[1,:] - c_vectors[0,:] #assuming at least 2 samples
        v_vecs = torch.clone(v).reshape(1,-1)

        #u = c_vectors - sample_means
        for t in range(c_vectors.shape[0]):
            if t == 0:
                # from second sample onwards
                continue 
                #sample_mean = torch.zeros(c_vectors[0,:].shape)#c_vectors[0, :]
            else:
                sample_mean = torch.mean(c_vectors[:t, :], dim = 0)

            u = (c_vectors[t,:] - sample_mean)
            u_vecs = torch.cat([u_vecs, u.reshape(1,-1)], dim=0)

            v_1_t = u
            v = v_1_t
            #compute v_(t-1)_t
            for k in range(1, t):
                gamma_k = self.rho * k/(k+1)
                num =  gamma_k *(torch.dot(v_vecs[k, :], u)) * v_vecs[k,:]
                den = 1 + gamma_k * (torch.dot(v_vecs[k,:], u_vecs[k,:]))
                v -= num/den 
            v_vecs = torch.cat([v_vecs, v.reshape(1, -1)], dim=0)

            #update delta
            uv = torch.dot(u, v)
            gamma_t = self.rho * (num_samples-1)/num_samples

            diff_fact_num =  gamma_t*(num_samples*torch.dot(u, delta_sim) - uv)
            diff_fact_den = 1+gamma_t*(uv)

            delta_sim = delta_sim - (1+diff_fact_num/diff_fact_den)*v/num_samples
        
        rhot = 1/(1+(num_samples - 1)*self.rho)
        return delta_sim/rhot

    def global_opt(self, g_delta):
        self.global_optimizer.zero_grad()

        #set global optimizer grad
        torch.nn.utils.vector_to_parameters(g_delta, self.base_net.parameters())

        #copy gradient data over to global net
        for p, g in zip(self.global_train.net.parameters(), self.base_net.parameters()):
            #print("p parmaeter ", p)
            #print("p grad paramter", p.grad)
            #print("p grad param.data ", p.grad.data)
            p.grad = g.data

        #update
        self.global_optimizer.step()

        return 

    def global_update_step(self):
        deltas = []

        #train client models/sample
        for client_num in range(self.num_clients):
            self.local_train(client_num)
            delta = self.local_delta(client_num)
            deltas.append(delta)
        deltas = torch.stack(deltas, dim=0)
        
        #global gradient
        g_delta = torch.mean(deltas, dim= 0)
        
        #take optimizer step in direction of g_delta for global net
        self.global_opt(g_delta)
        return 
    
    #when clients are already trained
    def global_update_step_trained_clients(self):
        deltas = []

        #train client models/sample
        for client_num in range(self.num_clients):
            delta = self.local_delta(client_num)
            deltas.append(delta)
        deltas = torch.stack(deltas, dim=0)
        
        #global gradient
        g_delta = torch.mean(deltas, dim= 0)
        
        #take optimizer step in direction of g_delta for global net
        self.global_opt(g_delta)
        return 
        
    
    def global_to_clients(self):
        for c in range(self.num_clients):
            self.client_train[c].net.load_state_dict(self.global_train.net.state_dict())
    
    def get_acc(self, net, valloader):
        print("Task ", self.task)
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)


    def train(self, valloader):
        acc = self.get_acc(self.global_train.net, valloader)
        
        for i in range(self.num_rounds):
            self.global_update_step()
            self.global_to_clients()
            acc = self.get_acc(self.global_train.net, valloader) #self.global_train.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

        #for reference, check client accuracies
        for c in range(self.num_clients):
            acc_c = self.client_train[c].test_acc(valloader)
            utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return

class ONESHOT_FL_CS:
    def __init__(self, num_clients, base_net,
                 traindata, distill_data,
                 num_rounds,
                 hyperparams, device, args,logger, non_iid=0.0, task="classify"):
        self.logger = logger
        self.all_data = traindata
        self.args = args
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.device = device
        self.num_clients = num_clients
        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.kdlr = hyperparams['kd_lr']
        self.kdopt = hyperparams['kd_optim_type']
        self.kdepoch = hyperparams['kd_epochs']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.optim_type = hyperparams['optim_type']
        self.seed = hyperparams['seed']

        self.model_save_dir = hyperparams['model_save_dir']

        # initialize nets and data for all clients
        self.client_nets = []
        self.client_nets2 = []
        self.optimizers = []
        self.optimizers2 = []

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset=traindata,
                                                                                   num_clients=num_clients,
                                                                                   client_data_size=(
                                                                                               self.datasize // num_clients),
                                                                                   batch_size=self.batch_size,
                                                                                   shuffle=False, non_iid_frac=non_iid,
                                                                                   outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            self.client_nets2.append(copy.deepcopy(base_net))

            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr, momentum=0.9))
                self.optimizers2.append(torch.optim.SGD(self.client_nets2[c].parameters(), lr=self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
                self.optimizers2.append(torch.optim.SGD(self.client_nets2[c].parameters(), lr=self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr=self.lr))
                self.optimizers2.append(torch.optim.Adam(self.client_nets2[c].parameters(), lr=self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
                self.optimizers2.append(torch.optim.SGD(self.client_nets2[c].parameters(), lr=self.lr))

        self.num_rounds = num_rounds

        self.task = task
        self.onemodel = True #False
        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.base_net = base_net


        distill_loader = torch.utils.data.DataLoader(distill_data,
                                                batch_size=self.batch_size, shuffle=True,
                                                pin_memory=True)
        self.student = copy.copy(base_net)
        self.distill = kd.KD(teacher=self,
                             student=self.student, lr=self.kdlr,
                             device=self.device,
                             train_loader=distill_loader,
                             kd_optim_type = self.kdopt
                             )

    def local_train(self, client_num):
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets2[client_num], loss = train_nets.sgd_train_step(net=self.client_nets2[client_num],
                                                                       optimizer=self.optimizers2[client_num],
                                                                       criterion=self.criterion,
                                                                       trainloader=c_dataloader, device=self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

    '''
    self.client_nets[client_num], loss = train_nets.sgd_train_step(net=self.client_nets[client_num],
                                                                       optimizer=self.optimizers[client_num],
                                                                       criterion=self.criterion,
                                                                trainloader=c_dataloader, device=self.device)
    '''


    # prediction on input x
    def predict_classify(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred_logit = self.client_nets[c](x)
            client_pred.append(pred_logit.view(pred_logit.size(0), -1).max(dim=-1).indices)
            if self.onemodel == False:
                self.client_nets2[c] = self.client_nets2[c].eval()
                pred_logit2 = self.client_nets2[c](x)
                client_pred.append(pred_logit2.view(pred_logit2.size(0), -1).max(dim=-1).indices)
        pred_class = torch.mode(torch.transpose(torch.stack(client_pred), 0, 1)).values
        pred_dist = torch.zeros(pred_class.size(0), pred_logit.size(1))
        for i in range(x.size(0)):
            pred_dist[i][pred_class[i]] = 1
        return pred_dist.to(self.device)

    def predict_regr(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred = self.client_nets[c](x)
            self.client_nets2[c] = self.client_nets2[c].eval()
            pred2 = self.client_nets2[c](x)
            client_pred.append(pred)
            client_pred.append(pred2)

        return torch.mean(torch.stack(client_pred), dim=0, keepdims=False).to(self.device)


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)
    def aggregate(self):
        self.distill.train(num_epochs = self.kdepoch)
        self.student = self.distill.student
        return

    def global_update_step(self):
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_train(client_num)
        self.aggregate()

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.predict(x)

            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            PATH =  "./results/models/" + self.args.dataset + "_fed_be_5_clients_1_rounds_log_{}_noniid_seed_".format(self.args.non_iid) +str(self.args.seed) + "_client_"+str(client_num)
            self.client_nets[client_num].load_state_dict(torch.load(PATH))
            if self.onemodel == False:
                for i in range(self.epoch_per_client):
                    self.local_train(client_num)
        self.aggregate()

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step_trained_clients()
            acc = self.distill.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                if self.onemodel == False:
                    acc_c2 = self.get_acc(self.client_nets2[c],valloader)
                    utils.print_and_log("Client {}, test accuracy: {} , {}".format(c, acc_c, acc_c2), self.logger)
                else:
                    utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)

        teacher_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}".format(i, teacher_acc), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl_cs", target_mode="teacher_oneshot_fl_cs"))

        return

class ONESHOT_FL:
    def __init__(self, num_clients, base_net,
                 traindata, distill_data,
                 num_rounds,
                 hyperparams, device, args,logger, non_iid=0.0, task="classify"):
        self.logger = logger
        self.all_data = traindata
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.device = device
        self.num_clients = num_clients
        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.kdlr = hyperparams['kd_lr']
        self.kdopt = hyperparams['kd_optim_type']
        self.kdepoch = hyperparams['kd_epochs']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.optim_type = hyperparams['optim_type']
        self.seed = hyperparams['seed']
        self.args = args
        # initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']
        self.model_save_dir = hyperparams['model_save_dir']

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset=traindata,
                                                                                   num_clients=num_clients,
                                                                                   client_data_size=(
                                                                                               self.datasize // num_clients),
                                                                                   batch_size=self.batch_size,
                                                                                   shuffle=False, non_iid_frac=non_iid,
                                                                                   outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))

            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr=self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))

        self.num_rounds = num_rounds

        self.task = task

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.base_net = base_net


        distill_loader = torch.utils.data.DataLoader(distill_data,
                                                batch_size=self.batch_size, shuffle=True,
                                                pin_memory=True)
        self.student = copy.copy(base_net)
        self.distill = kd.KD(teacher=self,
                             student=self.student, lr=self.kdlr,
                             device=self.device,
                             train_loader=distill_loader,
                             kd_optim_type = self.kdopt
                             )

    def local_train(self, client_num):
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net=self.client_nets[client_num],
                                                                       optimizer=self.optimizers[client_num],
                                                                       criterion=self.criterion,
                                                                       trainloader=c_dataloader, device=self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

    # prediction on input x
    def predict_classify(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred_logit = self.client_nets[c](x)

            client_pred.append(pred_logit)
        return torch.mean(torch.stack(client_pred), axis = 0)

    def predict_regr(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred = self.client_nets[c](x)
            client_pred.append(pred)

        return torch.mean(torch.stack(client_pred), dim=0, keepdims=False)


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)
    def aggregate(self):
        self.distill.set_student(self.client_nets[0].state_dict())

        #train the student via kd
        self.distill.train(num_epochs = self.kdepoch)
        self.student = self.distill.student
        return
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            PATH = "./results/models/" + self.args.dataset + "_fed_be_5_clients_1_rounds_log_{}_noniid_seed_".format(self.args.non_iid) +str(self.args.seed) + "_client_"+str(client_num)
            print(PATH)
            self.client_nets[client_num].load_state_dict(torch.load(PATH))
        self.aggregate()
    def global_update_step(self):
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_train(client_num)
        self.aggregate()

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc

    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.predict(x)

            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            #self.global_update_step()
            self.global_update_step_trained_clients()
            acc = self.distill.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}".format(i, teacher_acc), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"))
        #utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return

# requires base generative model AND base net/discriminative model
class GenBayes:
    def __init__(self, num_clients, base_net, base_gen, 
                 traindata, distill_data,
                 num_rounds,
                 hyperparams, device, args,logger, non_iid=0.0, task="classify"):
        self.logger = logger
        self.all_data = traindata
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.device = device
        self.num_clients = num_clients
        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.kdlr = hyperparams['kd_lr']
        self.kdopt = hyperparams['kd_optim_type']
        self.kdepoch = hyperparams['kd_epochs']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.optim_type = hyperparams['optim_type']
        self.seed = hyperparams['seed']
        self.args = args

        # initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []
        self.gen_opt = []

        #generative models on each client
        self.client_gens = []

        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']
        self.model_save_dir = hyperparams['model_save_dir']

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset=traindata,
                                                                                   num_clients=num_clients,
                                                                                   client_data_size=(
                                                                                               self.datasize // num_clients),
                                                                                   batch_size=self.batch_size,
                                                                                   shuffle=False, non_iid_frac=non_iid,
                                                                                   outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))
            self.client_gens.append(copy.deepcopy(base_gen))

            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr=self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))

            self.gen_opt.append(torch.optim.Adam(itertools.chain(self.client_gens[c].parameters()), lr = self.gen_lr, weight_decay = 1e-5))
        
        self.num_rounds = num_rounds

        self.task = task

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.base_net = base_net


        distill_loader = torch.utils.data.DataLoader(distill_data,
                                                batch_size=self.batch_size, shuffle=True,
                                                pin_memory=True)
        self.student = copy.copy(base_net)
        self.distill = kd.KD(teacher=self,
                             student=self.student, lr=self.kdlr,
                             device=self.device,
                             train_loader=distill_loader,
                             kd_optim_type = self.kdopt
                             )

    def local_train(self, client_num):
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net=self.client_nets[client_num],
                                                                       optimizer=self.optimizers[client_num],
                                                                       criterion=self.criterion,
                                                                       trainloader=c_dataloader, device=self.device)
        self.client_gens[client_num], gen_loss = nvp.gen_train_step(gen = self.client_gens[client_num],
                                                                        optimizer = self.gen_opt[client_num],
                                                                        trainloader = c_dataloader, device= self.device)

        print("Client  {}, p(y|x) Loss: {}, p(x) Loss: {}".format(client_num, loss, gen_loss))

    # prediction on input x
    def predict_classify(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred_logit = self.client_nets[c](x)

            client_pred.append(pred_logit)
        return torch.mean(torch.stack(client_pred), axis = 0)

    def predict_regr(self, x):
        client_pred = []
        for c in range(self.num_clients):
            self.client_nets[c] = self.client_nets[c].eval()
            pred = self.client_nets[c](x)
            client_pred.append(pred)

        return torch.mean(torch.stack(client_pred), dim=0, keepdims=False)


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)
   
    
    def aggregate(self):
        self.distill.set_student(self.client_nets[0].state_dict())

        #train the student via kd
        self.distill.train(num_epochs = self.kdepoch)
        self.student = self.distill.student
        return
   
   
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            PATH = "./results/models/" + self.args.dataset + "_fed_be_5_clients_1_rounds_log_{}_noniid_seed_".format(self.args.non_iid) +str(self.args.seed) + "_client_"+str(client_num)
            print(PATH)
            self.client_nets[client_num].load_state_dict(torch.load(PATH))
        self.aggregate()
    def global_update_step(self):
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_train(client_num)
        self.aggregate()

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc

    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.predict(x)

            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def train(self, valloader):
        for i in range(self.num_rounds):
            #self.global_update_step()
            self.global_update_step_trained_clients()
            acc = self.distill.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}".format(i, teacher_acc), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "oneshot_fl", target_mode="teacher_oneshot_fl"))
        #utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return



class FedBE:
    def __init__(self, num_clients, base_net,
                 traindata, distill_data,
                 num_rounds,
                 hyperparams, device, args,logger, non_iid=0.0, task="classify"):
        self.logger = logger
        self.all_data = traindata
        self.lr = hyperparams['lr']
        self.g_lr = hyperparams['g_lr']
        self.device = device
        self.num_clients = num_clients
        self.datasize = copy.deepcopy(hyperparams['datasize'])
        self.batch_size = hyperparams['batch_size']
        self.kdlr = hyperparams['kd_lr']
        self.kdopt = hyperparams['kd_optim_type']
        self.kdepoch = hyperparams['kd_epochs']
        self.epoch_per_client = hyperparams['epoch_per_client']
        self.outdim = hyperparams['outdim']
        self.optim_type = hyperparams['optim_type']
        self.seed = hyperparams['seed']
        self.args = args
        # initialize nets and data for all clients
        self.client_nets = []
        self.optimizers = []

        self.model_save_dir = hyperparams['model_save_dir']

        self.exp_id = hyperparams['exp_id']
        self.save_dir = hyperparams['save_dir']

        if non_iid > 0.0:
            self.client_dataloaders, self.client_datasize = datasets.non_iid_split(dataset=traindata,
                                                                                   num_clients=num_clients,
                                                                                   client_data_size=(
                                                                                               self.datasize // num_clients),
                                                                                   batch_size=self.batch_size,
                                                                                   shuffle=False, non_iid_frac=non_iid,
                                                                                   outdim=self.outdim)
        else:
            self.client_dataloaders, self.client_datasize = datasets.iid_split(traindata, num_clients, self.batch_size)

        for c in range(num_clients):
            self.client_nets.append(copy.deepcopy(base_net))

            if self.optim_type == "sgdm":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr, momentum=0.9))
            elif self.optim_type == "sgd":
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))
            elif self.optim_type == "adam":
                self.optimizers.append(torch.optim.Adam(self.client_nets[c].parameters(), lr=self.lr))
            else:
                utils.print_and_log("Optimizer type {} unkown, defualting to vanilla SGD")
                self.optimizers.append(torch.optim.SGD(self.client_nets[c].parameters(), lr=self.lr))

        self.num_rounds = num_rounds

        self.task = task

        if task == "classify":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.global_net = copy.deepcopy(base_net)
        self.base_net = base_net


        distill_loader = torch.utils.data.DataLoader(distill_data,
                                                batch_size=self.batch_size, shuffle=True,
                                                pin_memory=True)
        self.student = copy.copy(base_net)
        self.distill = kd.KD(teacher=self,
                             student=self.student, lr=self.kdlr,
                             device=self.device,
                             train_loader=distill_loader,
                             kd_optim_type = self.kdopt
                             )
        
        #posterior samples
        self.post_samples = []
        self.num_samples = 10 #set to number of clients (so we sample an equivalent number of clients)?

    def local_train(self, client_num):
        c_dataloader = self.client_dataloaders[client_num]

        self.client_nets[client_num], loss = train_nets.sgd_train_step(net=self.client_nets[client_num],
                                                                       optimizer=self.optimizers[client_num],
                                                                       criterion=self.criterion,
                                                                       trainloader=c_dataloader, device=self.device)

        print("Client {}, Loss: {}".format(client_num, loss))

    #####################################
    #  FOR ensemble inference
    ####################################

    #returns list of client nets as vectors
    def get_client_nets_as_vec(self):
        c_vectors = []
        for c in range(self.num_clients):
            c_vec = torch.nn.utils.parameters_to_vector(self.client_nets[c].parameters())
            c_vectors.append(c_vec)
        c_vectors = torch.stack(c_vectors, dim=0)

        return c_vectors

    def get_client_mean_cov(self):
        c_vectors = self.get_client_nets_as_vec()
        mean = torch.mean(c_vectors, dim=0)
        
        #too memory intensive - approximate with diagonal matrix
        #cov = torch.Tensor(np.cov((c_vectors).detach().numpy().T))
        
        #cov matrix has this vector as diagonal
        cov = torch.mean((c_vectors - mean)**2, dim = 0)

        return mean, cov

    def vec_to_net(self,vec):
        net = copy.deepcopy(self.base_net)
        torch.nn.utils.vector_to_parameters(vec, net.parameters())
        return net

    #calculates mean and variance of posterior, based
    #on client models
    def get_post_samples(self):
        #posterior samples = {client nets, mean net, sampled gaussian nets}
        
        #reset list of posterior samples
        self.post_samples = []

        #1 - client nets
        for c in range(self.num_clients):
            self.post_samples.append(copy.deepcopy(self.client_nets[c]))

        #2 - mean net
        mean, cov = self.get_client_mean_cov()

        self.post_samples.append(self.vec_to_net(mean))

        #3 - sampled gaussian nets
        dist = torch.distributions.Normal(mean, cov.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)
        #dist = torch.distributions.MultivariateNormal(loc = global_mean, precision_matrix=global_prec)
        global_samples = dist.sample([self.num_samples])
 
        
        for s in range(self.num_samples):
            sample = global_samples[s,0,:]
            self.post_samples.append(self.vec_to_net(sample))



    # prediction on input x
    def predict_classify(self, x):
        preds = []
        for m in range(len(self.post_samples)):
            self.post_samples[m] = self.post_samples[m].eval()
            pred_logit = self.post_samples[m](x)

            #append p(y|x, m_i) - for that model m_i
            preds.append(torch.nn.functional.softmax(pred_logit, dim=1))
        return torch.mean(torch.stack(preds), axis = 0)

    def predict_regr(self, x):
        preds = []
        for m in range(len(self.post_samples)):
            self.post_samples[m] = self.post_samples[m].eval()
            pred = self.post_samples[m](x)
            preds.append(pred)

        #mean of p(y|x) is avg of means
        return torch.mean(torch.stack(preds), dim=0, keepdims=False)


    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def aggregate(self):
        #get new set of posterior samples
        self.get_post_samples()

        self.distill.set_student(self.client_nets[0].state_dict())

        #train the student via kd
        self.distill.train(num_epochs = self.kdepoch)
        self.student = self.distill.student
        return
    ######################################################################################

    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            PATH = self.args.dataset + "_fed_sgd_5_clients_1_rounds_sgdm_optim_log_0.0_noniid_seed_"+str(self.args.seed) + "_client_"+str(client_num)
            print(PATH)
            self.client_nets[client_num].load_state_dict(torch.load(PATH))
        self.aggregate()

    def global_update_step(self):
        for client_num in range(self.num_clients):
            for i in range(self.epoch_per_client):
                self.local_train(client_num)
        self.aggregate()

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)

            _, pred_class = torch.max(pred, 1)

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100 * correct / total
        print("Accuracy on test set: ", acc)
        return acc

    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.predict(x)

            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    def get_acc(self, net, valloader):
        if self.task == "classify":
            return utils.classify_acc(net, valloader)
        else:
            return utils.regr_acc(net, valloader)

    def save_models(self):
        save_dir = self.model_save_dir
        save_name = self.exp_id + "_seed_"+str(self.seed)
        c_save = save_dir + "/"+save_name 

        utils.makedirs(save_dir)

        for c in range(self.num_clients):
            path = c_save + "_client_" + str(c)  
            torch.save(self.client_nets[c].state_dict(), path)

    def train(self, valloader):
        for i in range(self.num_rounds):
            self.global_update_step()
            #self.global_update_step_trained_clients()
            acc = self.distill.test_acc(valloader)
            utils.print_and_log("Global rounds completed: {}, test_acc: {}".format(i, acc), self.logger)

            #save client models (the global model is not sent to clients in this version, it is for 1 round only)
            self.save_models()

            for c in range(self.num_clients):
                acc_c = self.get_acc(self.client_nets[c],valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict_to_file(result = acc, seed = self.seed, file_name = self.save_dir + self.exp_id)

        #teacher acc
        teacher_acc = self.test_acc(valloader)
        utils.print_and_log("Global rounds completed: {}, Teacher test_acc: {}".format(i, teacher_acc), self.logger)
        utils.write_result_dict_to_file(result = teacher_acc, seed = self.seed, 
                                        file_name = self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "fed_be", target_mode="teacher_fed_be"))
        #utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger)
        return



# Implements DPredBayes (ours)
# same as F_MCMC but an additional distillation step is added
# Also runs evaluation for FMCMC distill, EP MCMC, and FedPA (1 round) using the same samples
class Calibrated_PredBayes_distill(EP_MCMC):
    def __init__(self, num_clients, base_net, 
                traindata, distill_data,
                num_rounds, hyperparams, 
                device, logger, non_iid = False, task = "classify"):
        EP_MCMC.__init__(self, num_clients, base_net, 
                traindata, 
                num_rounds, hyperparams, device, logger, non_iid, task)

        #make distillation dataset out of distill_data
        distill_loader = torch.utils.data.DataLoader(distill_data, 
                                                batch_size=self.batch_size, shuffle=True, 
                                                pin_memory=True)

        if task == "classify":
            self.student = copy.copy(base_net)
        else:
            #student model for regression needs to output variance as well
            self.student = models.LinearNetVar(inp_dim = base_net.input_dim, 
                                               num_hidden = base_net.num_hidden, 
                                               out_dim = base_net.out_dim)
            self.student = self.student.to(self.device)

        self.kd_optim_type = hyperparams['kd_optim_type']
        self.kd_lr = hyperparams['kd_lr']
        self.kd_epochs = hyperparams['kd_epochs']

        self.distill = kd.KD(teacher = self, 
                             student = self.student, lr = self.kd_lr,#5e-3
                             device = self.device,
                             train_loader = distill_loader,
                             kd_optim_type = self.kd_optim_type
                            )

        #in this experiment run, we also want to compare to EP MCMC and FedPA with same MCMC samples
        # so we setup this here
        self.fed_pa_trainer = FedPA(num_clients = num_clients, base_net = base_net, 
                                    traindata = traindata, 
                                    num_rounds = 1, 
                                    hyperparams = hyperparams, logger=None, non_iid = non_iid, task = self.task,
                                    device= self.device)

        #for training the interpolation param on distillation set
        self.interp_param = torch.tensor([hyperparams['init_interp_param']], device=self.device)
        self.interp_param.requires_grad = True 
        
        #self.interp_param_optim = torch.optim.LBFGS([self.interp_param], lr = hyperparams['interp_param_lr'], max_iter=50)
        #self.interp_param_optim = torch.optim.SGD([self.interp_param], lr=hyperparams['interp_param_lr'], momentum=0.1)
        self.interp_param_optim = torch.optim.Adam([self.interp_param], lr = hyperparams['interp_param_lr'])
    #Step 5
    # distill step - on unlabeled dataset, train student model to match 
    # the teacher predictions
    def aggregate(self):
        #train interpolation parameter
        self.train_interp(num_epochs = 10)

        #self.train_interp_lfbgs(num_epochs=10)

        #initialize student to a client network
        if self.task == "classify":
            self.distill.set_student(self.client_train[0].sampled_nets[-1])
        #else: the student is randomly initialized 
        
        #train the student via kd
        self.distill.train(num_epochs = self.kd_epochs) #kd_epochs = 50
        self.student = self.distill.student

        return
    
    def train_interp(self, num_epochs):
        #torch.autograd.set_detect_anomaly(True)
        #self.interp_param = self.interp_param.to(self.device)
        
        for i in range(num_epochs):
            epoch_loss = 0.0
            count = 0
            for x, y in self.distill.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.interp_param_optim.zero_grad()

                #need to do log softmax to ensure the exp of this is normalized

                if self.task == "classify":
                    pred_logits = self.predict_classify(x) #should be predicted log prob over classes
                   
                else:
                    pred_mean, pred_var = self.predict_regr(x) # should get mean and var of regression prediction 

                    #reshape to [*, 1] if one dimensional
                    #if len(pred_logits.shape) == 1:
                    #    pred_logits = pred_logits.unsqueeze(-1)

                #the loss is the negative log likelihood
                if self.task == "classify":
                    #print("Pred logits SHAPE: ", pred_logits.shape)
                    #print("y shape: ", y.shape)
                    loss = torch.nn.CrossEntropyLoss()(torch.log(pred_logits), y)
                else:
                    #print("pred mean: ", pred_mean)
                    #print("pred_var: ", pred_var)
                    loss = torch.nn.GaussianNLLLoss()(pred_mean, y, pred_var) #should be input, target, var
                    #loss = (pred_mean -  y)**2/(2*pred_std**2) + torch.log(torch.sqrt(2*np.pi)*pred_std)
                #self.interp_param.retain_grad() #for some reason, its a leaf node, so needs to retain grad
                
                #print("pred logits: ", pred_logits[75,:])
                #print("Loss: ", loss)
                
                loss.backward()

                #print("\nInterp param before update: ", self.interp_param)
                #print("Gradient of interp param:, ", self.interp_param.grad)
                #print("Requires gradient? : ", self.interp_param.requires_grad)
                #print("Is leaf? ", self.interp_param.is_leaf)
                
                self.interp_param_optim.step()
                
                #print("Interp param after step: ", self.interp_param)    
            

                epoch_loss += loss.item()
                
                count+=1
            print("Epoch: ", i+1, "Loss: ", epoch_loss, "Interp Param: ", self.interp_param)

            if (i+1)%20 == 0:
                self.test_acc(self.distill_train_loader)

    
        print("Training Interp Param Done! Interp Param: {}".format(self.interp_param))
        return 
    

    
    def train_interp_lfbgs(self, num_epochs):
        # do a pass thru dataset and collect product and mixture probabilities
        prod_list = []
        mix_list = []
        labels_list = []
            
        for x, y in self.distill.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
                
            prod_probs, mix_probs = self.get_prod_mix_predict_classify(x)

            prod_list.append(prod_probs)
            mix_list.append(mix_probs)
            labels_list.append(y)
        prods = torch.cat(prod_list).to(self.device)
        mixs = torch.cat(mix_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)
        

                
        def closure_eval():
            self.interp_param_optim.zero_grad()

            #need to do log softmax to ensure the exp of this is normalized
            if self.task == "classify":
                log_global_pred = self.interp_param*torch.log(prods) + (1- self.interp_param)*torch.log(mixs)
        
                # try temp scaling
                #log_global_pred = (1/self.interp_param) * torch.log(prods)
                  

            #the loss is the negative log likelihood
            if self.task == "classify":
                loss = torch.nn.CrossEntropyLoss()(log_global_pred, labels)
            #else:  
                #loss = torch.nn.GaussianNLLLoss()(pred_mean, y, pred_var) #should be input, target, var
                
            loss.backward()

            return loss 
                
        self.interp_param_optim.step(closure_eval)
        print("Training Interp Param Done! Interp Param: {}".format(self.interp_param))
        return 
    

    #so we can compare with EP MCMC 
    def ep_mcmc_aggregate(self):
        #in aggregation step - average all models

        global_prec = 0.0
        global_mean = 0.0

        #average over clients    
        for c in range(self.num_clients):
            mean, cov = self.get_client_sample_mean_cov(c)
            # in line below, either do this (for identity approx to covar), or 1/cov (for diagonal approx), former gave better results
            client_prec = 1 ############# 1/cov 
            global_prec += client_prec
            global_mean += client_prec * mean 

        global_mean = (1/global_prec) * global_mean 
        global_var = (1/global_prec)*torch.ones_like(global_mean).to(self.device)

        dist = torch.distributions.Normal(global_mean, global_var.reshape(1, -1))
        dist = torch.distributions.independent.Independent(dist, 1)
        
        global_samples = dist.sample([self.num_g_samples])
      
    
        self.global_train.sampled_nets = []
        
        for s in range(self.num_g_samples):
            # in line below, can either use only mean as sample, or sample from gaussian centered at mean
            # from experiments, just the mean works better (possibly due to high correlation between parameters that a diagonal approx can't capture)
            sample = global_mean #global_samples[s,0,:] ####################################### 
            

            #load into global net
            torch.nn.utils.vector_to_parameters(sample, self.global_train.net.parameters())
            self.global_train.sampled_nets.append(copy.deepcopy(self.global_train.net.state_dict()))

        return

    def get_prod_mix_predict_classify(self, x):
        global_pred_product = 1.0
        global_pred_mixture = 0.0

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred_product *= pred
            global_pred_mixture += pred/(self.num_clients)
        
        
        global_pred_product = global_pred_product/torch.sum(global_pred_product, dim=-1, keepdims=True)
        global_pred_mixture = global_pred_mixture/torch.sum(global_pred_mixture, dim=-1, keepdims=True)

        global_pred_product_no_grad = global_pred_product.detach().clamp(min = 1e-41)
        global_pred_mixture_no_grad = global_pred_mixture.detach().clamp(min= 1e-41)

        #renormalize
        global_pred_product_no_grad = global_pred_product_no_grad/global_pred_product_no_grad.sum(dim=-1, keepdims=True)
        global_pred_mixture_no_grad = global_pred_mixture_no_grad/global_pred_mixture_no_grad.sum(dim=-1, keepdims=True)

        return global_pred_product_no_grad, global_pred_mixture_no_grad


    #prediction on input x
    def predict_classify(self, x):
        global_pred_product = 1.0
        global_pred_mixture = 0.0

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
                
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred = torch.mean(pred_list, dim=0, keepdims=False)
            
            #assuming a uniform posterior
            global_pred_product *= pred
            global_pred_mixture += pred/(self.num_clients)
        
        
        global_pred_product = global_pred_product/torch.sum(global_pred_product, dim=-1, keepdims=True)
        global_pred_mixture = global_pred_mixture/torch.sum(global_pred_mixture, dim=-1, keepdims=True)

        global_pred_product_no_grad = global_pred_product.detach().clamp(min = 1e-41)
        global_pred_mixture_no_grad = global_pred_mixture.detach().clamp(min= 1e-41)

        #renormalize
        global_pred_product_no_grad = global_pred_product_no_grad/global_pred_product_no_grad.sum(dim=-1, keepdims=True)
        global_pred_mixture_no_grad = global_pred_mixture_no_grad/global_pred_mixture_no_grad.sum(dim=-1, keepdims=True)
    

        #interpolate between the two distributions with self.interp_param
        log_global_pred = torch.clamp(self.interp_param, min=0.0)*torch.log(global_pred_product_no_grad) + torch.clamp((1- self.interp_param), min=0.0)*torch.log(global_pred_mixture_no_grad)
        
        # try temp scaling
        #log_global_pred = (1/self.interp_param) * torch.log(global_pred_product_no_grad)
        
        #log_global_pred = log_global_pred.clamp(min=-40)
        global_pred = torch.functional.F.softmax(log_global_pred, dim = -1)#torch.exp(log_global_pred)/torch.sum(torch.exp(log_global_pred), dim=-1, keepdims=True)
        global_pred = global_pred/torch.sum(global_pred, dim=-1, keepdims=True) 

        #print("\nInterp Param: ", self.interp_param)
        #print("Mixture Pred: ", global_pred_mixture)
        #print("Product Pred: ", global_pred_product)
        #print("Logit global pred: ", log_global_pred)
        #print("global_pred: ", global_pred)
        
        #if torch.isinf(-log_global_pred).any():
        #    print("\n\n IS Neg INF!!!!!!!\n\n")
        #    
        #    
        #   inf_idx = torch.nonzero(torch.isinf(-log_global_pred))
            
        #    print("Inf idx: ", inf_idx[0,0])
        #    print("Size of global pred: ", global_pred_mixture.shape)
        #    print("Mix index for nan: ", global_pred_mixture[inf_idx[0,0], :])
        #    print("Prod index for nan: ", global_pred_product[inf_idx[0,0], :])
        #    print("Log global pred at nan idx: ", log_global_pred[inf_idx[0,0], :])
        #    print("Global pred: ", global_pred[inf_idx[0,0], :])
        #    print("Taking log of global pred again: ", torch.log(global_pred[inf_idx[0,0], :]))
        #if torch.isnan(global_pred).any():
        #    print("\n\n IS NAN !!!!!!!\n\n")
        #    return
        #    _, nan_idx = torch.isnan(global_pred).max()
        #    
        #    print("Mix index for nan: ", global_pred_mixture[:, nan_idx])
        #    print("Prod index for nan: ", global_pred_product[:, nan_idx])
        #    print("Log global pred at nan idx: ", log_global_pred[:, nan_idx])
        #
        #    return 
        
        return global_pred 
    
    def predict_regr(self,x):
        global_pred_product = 0.0
        prec_sum_product = 0.0

        global_pred_mixture = 0.0 
        second_moment_mixture = 0.0 

        for c in range(self.num_clients):
            pred_list = self.client_train[c].ensemble_inf(x, out_probs=True)
            
            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            pred_mean = torch.mean(pred_list, dim=0, keepdims=False)

            # the regression model is assumed to make predictions as N(f(x), 1) where f(x) = neural net output
            #Therefore, the variance in the prediction is 1.0 (the homeoscedatic assumed variance) + bayesian variance
            if pred_list.shape[0] == 1:
                pred_var = torch.ones_like(pred_mean)
            else: 
                pred_var = 1.0 + torch.var(pred_list, dim = 0, keepdims = False)
            #print("client: {}, pred_mean: {}".format(c, pred_mean))
            #print("client: {}, pred_var: {}".format(c, pred_var))

            #product aggregation
            #assuming a uniform posterior
            global_pred_product += pred_mean/pred_var
            prec_sum_product += 1/pred_var
            
            # mixture aggregation
            global_pred_mixture += pred_mean/self.num_clients
            second_moment_mixture +=  (pred_var + pred_mean**2)/self.num_clients

        global_pred_product = global_pred_product/prec_sum_product
        var_product = 1/prec_sum_product 

        var_mixture = second_moment_mixture - (global_pred_mixture)**2

        #print("global_pred_product: ", global_pred_product)
        #print("var_product: ", var_product)
        #print("global_pred_mixture: ", global_pred_mixture)
        #print("var_mixture: ", var_mixture)

        # detach gradients (wrt interp_param these are all constants)
        global_pred_product = global_pred_product.detach()
        var_product = var_product.detach()
        global_pred_mixture = global_pred_mixture.detach()
        var_mixture = var_mixture.detach()


        #interpolate the mean and variance between the product and mixture 
        global_prec =  (self.interp_param)/var_product + (1 - self.interp_param)/var_mixture
        global_var = 1/global_prec 

        global_pred = (self.interp_param)*global_pred_product/var_product + (1-self.interp_param)*global_pred_mixture/var_mixture 
        global_pred = global_pred * global_var
        return global_pred, global_var #should output mean and variance !!!!!

    #Step 3 and 4 (local posterior estimate  + aggregation for global posterior)
    def predict(self, x):
        if self.task == "classify":
            return self.predict_classify(x)
        else:
            return self.predict_regr(x)

    def test_classify(self, testloader):
        total = 0
        correct = 0

        for batch_idx, (x, y) in enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.predict(x)
            _, pred_class = torch.max(pred, 1)    

            total += y.size(0)
            correct += (pred_class == y).sum().item()

        acc = 100*correct/total
        print("Accuracy on test set: ", acc)
        return acc
 
    def test_mse(self, testloader):
        total_loss = 0.0
        criterion = torch.nn.MSELoss()

        for batch_idx,(x,y) in  enumerate(testloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            pred, pred_var = self.predict(x)
           
            pred = pred.reshape(y.shape)
            total_loss += criterion(pred, y).item()

        print("MSE on test set: ", total_loss)
        return total_loss    

    def test_acc(self, testloader):
        if self.task == "classify":
            return self.test_classify(testloader)
        else:
            return self.test_mse(testloader)

    #load the local models trained previously, then tune beta and distill (call aggregate)
    def global_update_step_trained_clients(self):
        for client_num in range(self.num_clients):
            model_PATH =  "./results_distill_f_mcmc/models/" + self.dataset + "_mcmc_5_clients_1_rounds_log_{}_noniid_seed_".format(self.non_iid) +str(self.seed) + "_client_"+str(client_num) 

            # there should be 6 samples for each dataset
            for idx in range(self.max_samples): 
                path = model_PATH + "_sample_" + str(idx) 
                weight_dict = torch.load(path)
                self.client_train.sampled_nets.append(weight_dict)
            
        self.aggregate()

    def train(self, valloader):
        for i in range(self.num_rounds):
            # Step 1 and 2: local MCMC sampling + communicating to server
            
            ################################
            #train from scratch
            #self.global_update_step()

            #or load saved models
            self.global_update_step_trained_clients()
            ######################################


            acc = self.distill.test_acc(valloader)
            nllhd, cal_error = utils.test_calibration(model = self.distill.student, testloader=valloader, task=self.task, 
                                                      device = self.device, model_type= "single")

            utils.print_and_log("\nGlobal rounds completed: {}, distilled_f_mcmc test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(i, acc, nllhd, cal_error), self.logger)

            for c in range(self.num_clients):
                acc_c = self.client_train[c].test_acc(valloader)
                utils.print_and_log("Client {}, test accuracy: {}".format(c, acc_c), self.logger)

        utils.write_result_dict(result=acc, seed=self.seed, logger_file=self.logger, type="acc")
        utils.write_result_dict(result=nllhd, seed=self.seed, logger_file=self.logger, type="nllhd")
        utils.write_result_dict(result=cal_error, seed=self.seed, logger_file=self.logger, type="cal")

        #evaluate samples on other methods (FMCMC, and EP MCMC)
        f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("\n\nGlobal rounds completed: {}, f_mcmc test_acc: {}".format(i, f_mcmc_acc), self.logger)
        f_mcmc_nllhd, f_mcmc_cal_error = utils.test_calibration(model = self, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("Global rounds completed: {}, f_mcmc test_ NLLHD: {}, Cal Error: {}\n".format(i, f_mcmc_nllhd, f_mcmc_cal_error), self.logger)

        #save to dict
        utils.write_result_dict_to_file(result=f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=f_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=f_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="f_mcmc"),
                                 type = "cal")


        #evaluate f_mcmc_model without interpolation param = 1 (product) and 0 (mixture)
        self.tuned_interp_param =  torch.clone(self.interp_param)

        # PRODUCT model/ previous FMCMC
        self.interp_param = torch.tensor([1.0], device=self.device) 
        prod_f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("\nGlobal rounds completed: {}, PRODUCT f_mcmc test_acc: {}".format(i, prod_f_mcmc_acc), self.logger)
        prod_f_mcmc_nllhd, prod_f_mcmc_cal_error = utils.test_calibration(model = self, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("Global rounds completed: {}, PRODUCT f_mcmc test_ NLLHD: {}, Cal Error: {}\n".format(i, prod_f_mcmc_nllhd, prod_f_mcmc_cal_error), self.logger)

        #save to dict
        utils.write_result_dict_to_file(result=prod_f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="product_f_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=prod_f_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="product_f_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=prod_f_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="product_f_mcmc"),
                                 type = "cal")


        # MIXTURE model/ previous FMCMC
        self.interp_param = torch.tensor([0.0], device=self.device)
        mix_f_mcmc_acc = self.test_acc(valloader)
        utils.print_and_log("\nGlobal rounds completed: {}, MIXTURE f_mcmc test_acc: {}".format(i, mix_f_mcmc_acc), self.logger)
        mix_f_mcmc_nllhd, mix_f_mcmc_cal_error = utils.test_calibration(model = self, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("Global rounds completed: {}, MIXTURE f_mcmc test_ NLLHD: {}, Cal Error: {}\n".format(i, mix_f_mcmc_nllhd, mix_f_mcmc_cal_error), self.logger)

        #save to dict
        utils.write_result_dict_to_file(result=mix_f_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="mixture_f_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=mix_f_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="mixture_f_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=mix_f_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="mixture_f_mcmc"),
                                 type = "cal")

        #revert self interp param
        self.interp_param = self.tuned_interp_param

        #compute ep mcmc result and store in global_train
        self.ep_mcmc_aggregate()
        ep_mcmc_acc = self.global_train.test_acc(valloader)
        ep_mcmc_nllhd, ep_mcmc_cal_error = utils.test_calibration(model = self.global_train, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "ensemble")
        utils.print_and_log("\nGlobal rounds completed: {}, ep_mcmc test_acc: {}, NLLHD: {}, Calibration Error: {}\n".format(i, ep_mcmc_acc, ep_mcmc_nllhd, ep_mcmc_cal_error), self.logger)
        
        #save to dict 
        utils.write_result_dict_to_file(result=ep_mcmc_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=ep_mcmc_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=ep_mcmc_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="ep_mcmc"),
                                 type = "cal")


        #for fed_pa 1 round results
        #copy over trained clients
        self.fed_pa_trainer.client_train = copy.deepcopy(self.client_train)

        ##### UNCOMMENT LATER ###########
        #take step of FedPA 
        self.fed_pa_trainer.global_update_step_trained_clients()
        fed_pa_acc = self.fed_pa_trainer.get_acc(self.fed_pa_trainer.global_train.net, valloader)
        fed_pa_nllhd, fed_pa_cal_error = utils.test_calibration(model = self.fed_pa_trainer.global_train.net, testloader=valloader, task=self.task,
                                                      device = self.device, model_type= "single")
        utils.print_and_log("Global rounds completed: {}, fed_pa test_acc: {},  NLLHD: {}, Calibration Error: {}".format(i, fed_pa_acc, fed_pa_nllhd, fed_pa_cal_error), self.logger)
        
        #save to dict 
        utils.write_result_dict_to_file(result=fed_pa_acc, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"),
                                 type = "acc")
        utils.write_result_dict_to_file(result=fed_pa_nllhd, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"),
                                 type = "nllhd")
        utils.write_result_dict_to_file(result=fed_pa_cal_error, seed = self.seed, 
                                 file_name= self.save_dir + utils.change_exp_id(exp_id_src=self.exp_id, source_mode = "distill_f_mcmc", target_mode="fed_pa"),
                                 type = "cal")



        #save client sample models 
        #self.save_models()

        return
