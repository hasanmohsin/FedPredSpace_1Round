import torch
import numpy as np
import swa
import copy
import torch.nn.functional as F

class KD:
    def __init__(self, teacher, student, lr, device, train_loader):
        self.teacher = teacher
        
        self.student = student

        #trying student as existing trained model
        #print("Num nets ", len(teacher.client_train[0].sampled_nets))
        #self.student = copy.copy(teacher.client_train[0].sampled_nets[5])

        self.lr = lr
        
        self.optimizer = torch.optim.Adam(params = student.parameters(), lr = 1e-4)
        #base_opt = torch.optim.SGD(self.student.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00001)
        #self.optimizer = swa.SWA(base_opt, swa_start=100, swa_freq=10, swa_lr=None)

        self.train_loader = train_loader

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
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
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

