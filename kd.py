import torch
import numpy as np
import swa
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import utils 

class KD:
    def __init__(self, teacher, student, lr, device, train_loader, kd_optim_type = "sgdm"):
        self.teacher = teacher
        
        self.student = student

        #trying student as existing trained model
        #print("Num nets ", len(teacher.client_train[0].sampled_nets))
        #self.student = copy.copy(teacher.client_train[0].sampled_nets[5])

        self.lr = lr
        self.kd_optim_type = kd_optim_type
        
        if kd_optim_type == "adam":
            self.optimizer = torch.optim.Adam(params = self.student.parameters(), lr = self.lr)#, weight_decay=0.00001)
        elif kd_optim_type == "sgdm":
            self.optimizer = torch.optim.SGD(self.student.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.00001)
        elif kd_optim_type == "swa":
            base_opt = torch.optim.SGD(self.student.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.00001)
            self.optimizer = swa.SWA(base_opt, swa_start=100, swa_freq=10, swa_lr=None)

        self.train_loader = train_loader

        self.task = self.teacher.task

        if self.teacher.task == "classify":
            self.criterion = torch.nn.KLDivLoss(reduction = "batchmean")
            print("Classification task: using KL div loss")
        else:
            self.criterion = torch.nn.MSELoss()
            #print("Regression Task: using MSE loss")
            
            #self.criterion = torch.nn.GaussianNLLLoss()
            print("Regression Task: using KL Div between Gaussians as loss (and MSE as criterion)")

        self.device = device
        self.task = self.teacher.task

    def set_student(self, student_targ_dict):
        self.student.load_state_dict(copy.copy(student_targ_dict))
        return

    def train_step(self):
        epoch_loss = 0.0
        count = 0
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()

            #need to do log softmax to ensure the exp of this is normalized

            if self.task == "classify":
                pred_logits =  F.log_softmax(self.student(x), dim=-1)
            else:
                pred_mean, pred_var = self.student(x)

                #reshape to [*, 1] if one dimensional
                if len(pred_mean.shape) == 1:
                    pred_mean = pred_mean.unsqueeze(-1)
                    pred_var = pred_var.unsqueeze(-1)

            with torch.no_grad():
                if self.task == "classify":
                    teach_targ = self.teacher.predict(x)
                else:
                    teach_targ, teach_var = self.teacher.predict(x)

            #print("x size: {}".format(x.shape))
            #print("Student out size: {}".format(pred_logits.size()))
            #print("Teacher out size: {}".format(teach_targ.size()))

            if self.task == "classify":
                #print("teach_targ.shape: ", teach_targ.shape) # should be B (batch) x C (num classes)
                #print("teach_targ sum: ", teach_targ.sum(dim =-1))
                #print("Pred logits shape: ", pred_logits.shape) 
                #print("Pred logits exp sum: ", pred_logits.exp().sum(dim =-1))
            
                pred_logits = pred_logits.reshape(teach_targ.shape) # reshape like teacher predictions 
            elif self.task == "regression":
                pred_mean = pred_mean.reshape(teach_targ.shape)

                #print("pred_var: ", pred_var)
                #print("pred_var shape: ", pred_var.shape)
                pred_var = pred_var.reshape(teach_var.shape)

            #compute loss
            if self.task == "classify":
                loss = self.criterion(pred_logits, teach_targ.detach())
            else:
                loss = utils.kl_div_gauss(pred_mean, pred_var, teach_targ, teach_var)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            count+=1
        return epoch_loss
    

    # training with regular SGD
    def train(self, num_epochs):
        self.student = self.student.train()

        for i in range(num_epochs):
            epoch_loss = self.train_step()
            print("Epoch: ", i+1, "Loss: ", epoch_loss)
            
            if (i+1)%20 == 0:
                self.test_acc(self.train_loader)
        
        print("Training Done!")
        return

    def test_acc(self, testloader):
        #for classification
        total = 0
        t_correct = 0
        s_correct = 0

        #FOR REGRESSION tasks
        total_s_loss  = 0.0
        total_t_loss = 0.0

        for batch_idx, (x, y) in enumerate(testloader):
            x  = x.to(self.device)
            y = y.to(self.device)

            if self.task == "classify":
                t_pred = self.teacher.predict(x)
            else:
                t_pred, t_var = self.teacher.predict(x) 

            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            #t_pred = torch.mean(teach_pred_list, dim=0, keepdims=False)
            if self.task == "classify":
                s_pred = self.student(x)
            else:
                s_pred, s_var = self.student(x)
                

            if self.task == "classify":
                _, t_pred_class = torch.max(t_pred, 1)    
                _, s_pred_class = torch.max(s_pred , 1)

                total += y.size(0)
                t_correct += (t_pred_class == y).sum().item()
                s_correct += (s_pred_class == y).sum().item()
            else:
                t_pred = t_pred.reshape(y.shape)
                s_pred = s_pred.reshape(y.shape)

                s_loss = self.criterion(s_pred, y)
                t_loss = self.criterion(t_pred, y)

                total_s_loss += s_loss.item()
                total_t_loss += t_loss.item()

        if self.task == "classify":
            t_acc = 100*t_correct/total
            print("Teacher accuracy on test set: ", t_acc)
            s_acc = 100*s_correct/total
            print("Student accuracy on test set: ", s_acc)
            return s_acc
        else:
            print("Teacher MSE on test set: ", total_t_loss)
            print("Average teacher var: {}".format(t_var.mean()))
            print("Student MSE on test set: ", total_s_loss)
            print("Average student var: {}".format(s_var.mean()))
            return total_s_loss



#KD with adversarially generated data points
class KD_gen_data:
    def __init__(self, teacher, student, lr, device, train_loader, inp_dim, synth_data_size = 15000):
        self.teacher = teacher
        
        self.student = student

        self.lr = lr
        
        self.optimizer = torch.optim.Adam(params = student.parameters(), lr = 1e-4)#, weight_decay=0.00001)

        #for test set
        self.train_loader = train_loader

        self.synth_data_size =15000
        self.batch_size = inp_dim[0]
        self.gen_per_step = int(self.synth_data_size/self.batch_size)
        self.inp_dim = inp_dim

        self.task = self.teacher.task

        if self.teacher.task == "classify":
            self.criterion = torch.nn.KLDivLoss(reduction = "batchmean")
            print("Classification task: using KL div loss")
        else:
            self.criterion = torch.nn.MSELoss()
            print("Regression Task: using MSE loss")

        self.device = device
        self.task = self.teacher.task

    def set_student(self, student_targ_dict):
        self.student.load_state_dict(copy.copy(student_targ_dict))
        return

    def train_step(self):
        epoch_loss = 0.0
        count = 0
        x = None 

        for i in range(self.gen_per_step):
            
            #generate BATCH_SIZE number of inputs
            #print("### GENERATING DATA ###")
            #generate ylabel
            ylabel = torch.randint(low=0, high = 10, size = [self.batch_size]).to(self.device)

            if x is None:
                start_point = None
                #start_point, ylabel = next(iter(self.train_loader)) #try starting from a known datapoint
                #start_point = start_point.to(self.device)
                #ylabel = ylabel.to(self.device)
            else:
                start_point = None
                #inter = x + torch.randn(self.inp_dim, device = self.device)
                #start_point = copy.copy(inter.detach())
                #start_point.requires_grad= True
                #ylabel = torch.randint(low=0, high = 10, size = [self.batch_size]).to(self.device)
            
            x = self.gen_data_llhd(start_point=start_point, ylabel = ylabel)
            #print("### DONE GENERATING DATA ###")

            #save images 
            if i%50 == 0:
                im0 = x[0,0].detach().data.cpu().numpy()
                label0 = ylabel[0]
                name0 = "im0_iter_{}_label_{}".format(i, label0)
                
                plt.imshow(im0)
                plt.title("Im 0, Iteration (in Epoch): {}, Y Label: {}".format(i, label0))
                plt.savefig("./gen_data_imgs/{}".format(name0),  bbox_inches = "tight", pad_inches = 0.0)

                im12 = x[12,0].detach().data.cpu().numpy()
                label12 = ylabel[12]
                name12 = "im12_iter_{}_label_{}".format(i, label12)

                plt.imshow(im12)
                plt.title("Im 12, Iteration (in Epoch): {}, Y Label: {}".format(i, label12))
                plt.savefig("./gen_data_imgs/{}".format(name12),  bbox_inches = "tight", pad_inches = 0.0)

            self.optimizer.zero_grad()

            #need to do log softmax to ensure the exp of this is normalized

            if self.task == "classify":
                pred_logits =  F.log_softmax(self.student(x), dim=-1)
            else:
                pred_logits = self.student(x)

                #reshape to [*, 1] if one dimensional
                if len(pred_logits.shape) == 1:
                    pred_logits = pred_logits.unsqueeze(-1)

            with torch.no_grad():
                teach_targ = self.teacher.predict(x)

            #print("x size: {}".format(x.shape))
            #print("Student out size: {}".format(pred_logits.size()))
            #print("Teaher out size: {}".format(teach_targ.size()))

            loss = self.criterion(pred_logits, teach_targ)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            count+=1
        return epoch_loss
    
    #adversarially generate data
    def gen_data(self, start_point = None):
        #inp_dim should contain batch size
        #random normal init

        if start_point is None:
            x = torch.randn(self.inp_dim, requires_grad=True, device = self.device)
        else:
            x = start_point
        #x.requires_grad = True
        #x = x.to(self.device)
        
        inp_opt = torch.optim.Adam(params = [x], lr = 1e-3)

        gen_steps = 1#50

        for i in range(gen_steps):
            inp_opt.zero_grad()

            if self.task == "classify":
                pred_logits =  F.log_softmax(self.student(x), dim=-1)
            else:
                pred_logits = self.student(x)

                #reshape to [*, 1] if one dimensional
                if len(pred_logits.shape) == 1:
                    pred_logits = pred_logits.unsqueeze(-1)

            with torch.no_grad():   
                teach_targ = self.teacher.predict(x)

            #maximize the loss wrt x
            loss = self.criterion(pred_logits, teach_targ)
            neg_loss = -1*loss
            neg_loss.backward()
            inp_opt.step()

            #print("Iteration {}, Generation (KL) Loss to max: {}".format(i, neg_loss))

        x.requires_grad = False

        return x

    #adversarially generate data
    #ylabel is a batch of targets, generated uniformly
    def gen_data_llhd(self, ylabel, start_point = None):
        #inp_dim should contain batch size
        #random normal init

        if start_point is None:
            x = torch.randn(self.inp_dim, requires_grad=True, device = self.device)
        else:
            #perturb = torch.randn(self.inp_dim, requires_grad=True, device = self.device)
            x = start_point
            
            #start_point.requires_grad = True
            #x = start_point + perturb
            #x.requires_grad = True
        #x = x.to(self.device)
        
        inp_opt = torch.optim.Adam(params = [x], lr = 1e-3)

        gen_steps = 50
        
        loss_fn = torch.nn.CrossEntropyLoss()

        for i in range(gen_steps):
            inp_opt.zero_grad()

            if self.task == "classify":
                pred_logits =  F.log_softmax(self.student(x), dim=-1)
            else:
                pred_logits = self.student(x)

                #reshape to [*, 1] if one dimensional
                if len(pred_logits.shape) == 1:
                    pred_logits = pred_logits.unsqueeze(-1)

            torch.autograd.set_detect_anomaly(True)
            teach_targ = self.teacher.client_train[-1].net(x) # ensemble_inf(x, out_probs=True).mean(dim=0, keepdims=False)

            #logit scores for this input

            #calculate teacher likelihood:
            # log p(ylabel | x_gen ,D)    
            #teacher_llhd = teach_targ[ylabel].log()
            
            #minimize the loss wrt x
            #average entropy of teacher predictions should be minimized (to get inputs for which teacher is confident)
            loss = loss_fn(teach_targ, ylabel)
            loss.backward()
            inp_opt.step()

            #print("Iteration {}, Average teacher entropy/confidence on x: {}".format(i, loss))

        x.requires_grad = False

        return x

    #adversarially generate data
    def gen_data_conf(self, start_point = None):
        #inp_dim should contain batch size
        #random normal init

        if start_point is None:
            x = torch.randn(self.inp_dim, requires_grad=True, device = self.device)
        else:
            x = start_point
        #x.requires_grad = True
        #x = x.to(self.device)
        
        inp_opt = torch.optim.Adam(params = [x], lr = 1e-3)

        gen_steps = 50

        for i in range(gen_steps):
            inp_opt.zero_grad()

            if self.task == "classify":
                pred_logits =  F.log_softmax(self.student(x), dim=-1)
            else:
                pred_logits = self.student(x)

                #reshape to [*, 1] if one dimensional
                if len(pred_logits.shape) == 1:
                    pred_logits = pred_logits.unsqueeze(-1)

            torch.autograd.set_detect_anomaly(True)
            teach_targ = self.teacher.client_train[0].net(x) # ensemble_inf(x, out_probs=True).mean(dim=0, keepdims=False)

            #dont need following line for ensemble inf
            teach_targ = torch.nn.functional.softmax(teach_targ, dim=1)

            #calculate teacher confidence:
            # can either do variance of sample outputs, or entropy of final p(y|x,D)    
            teacher_ent = torch.sum(-teach_targ * teach_targ.log(), axis=-1)

            #minimize the loss wrt x
            #average entropy of teacher predictions should be minimized (to get inputs for which teacher is confident)
            loss = teacher_ent.mean(axis=0)
            loss.backward()
            inp_opt.step()

            #print("Iteration {}, Average teacher entropy/confidence on x: {}".format(i, loss))

        x.requires_grad = False

        return x

    # training with regular SGD
    def train(self, num_epochs):
        self.student = self.student.train()

        for i in range(num_epochs):
            epoch_loss = self.train_step()
            print("Epoch: ", i+1, "Loss: ", epoch_loss)
            self.test_acc(self.train_loader)
        
        print("Training Done!")
        return

    def test_acc(self, testloader):
        #for classification
        total = 0
        t_correct = 0
        s_correct = 0

        #FOR REGRESSION tasks
        total_s_loss  = 0.0
        total_t_loss = 0.0

        for batch_idx, (x, y) in enumerate(testloader):
            x  = x.to(self.device)
            y = y.to(self.device)

            t_pred = self.teacher.predict(x)

            #average to get p(y | x, D)
            # shape: batch_size x output_dim
            #t_pred = torch.mean(teach_pred_list, dim=0, keepdims=False)
            s_pred = self.student(x)

            if self.task == "classify":
                _, t_pred_class = torch.max(t_pred, 1)    
                _, s_pred_class = torch.max(s_pred , 1)

                total += y.size(0)
                t_correct += (t_pred_class == y).sum().item()
                s_correct += (s_pred_class == y).sum().item()
            else:
                t_pred = t_pred.reshape(y.shape)
                s_pred = s_pred.reshape(y.shape)
                s_loss = self.criterion(s_pred, y)
                t_loss = self.criterion(t_pred, y)
                total_s_loss += s_loss.item()
                total_t_loss += t_loss.item()

        if self.task == "classify":
            t_acc = 100*t_correct/total
            print("Teacher accuracy on test set: ", t_acc)
            s_acc = 100*s_correct/total
            print("Student accuracy on test set: ", s_acc)
            return s_acc
        else:
            print("Teacher MSE on test set: ", total_t_loss)
            print("Student MSE on test set: ", total_s_loss)
            return total_s_loss

