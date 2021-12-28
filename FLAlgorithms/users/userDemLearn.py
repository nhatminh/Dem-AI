import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer, DemSGD, DemProx_SGD
from FLAlgorithms.users.userbase_dem import User
import copy

# Implementation for pFeMe clients
from utils.train_utils import KL_Loss


class UserDemLearn(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,L_k,
                 local_epochs, optimizer, K, personal_learning_rate, args):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, L_k,
                         local_epochs)

        # if(model[1] == "Mclr_CrossEntropy"):
        #     self.loss = nn.CrossEntropyLoss()
        # else:
        #     self.loss = nn.NLLLoss()

        self.loss = nn.CrossEntropyLoss()  ##  we can also use NLLLoss
        self.criterion_KL = KL_Loss(temperature=3.0)

        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.mu = args.mu

        # if(self.mu==0):
        #     print("Using DemSGD Optimizer")
        #     # self.SGD_optimizer = DemSGD(self.model.parameters(), lr=self.personal_learning_rate)
        #     self.SGD_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # else:
        print("Using ProxSGD Optimizer")
        self.SGD_optimizer = DemProx_SGD(self.model.parameters(), lr=self.learning_rate, mu=self.mu)

        # self.optimizer = Prox_SGD(self.model.parameters(), lr=self.personal_learning_rate, mu=self.mu)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        if (self.mu > 0):
            init_model = copy.deepcopy(self.model)

        for epoch in range(1, epochs + 1):  # local update
            self.model.train()
            # batch_idx=0
            for X,y in self.trainloader:
                # print(f"batch index{batch_idx}:")
                # batch_idx+= 1
                X, y = X.to(self.device), y.to(self.device)#self.get_next_train_batch()
                self.SGD_optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                if (self.mu == 0):
                    # updated_model, _ = self.SGD_optimizer.step()
                    _ = self.SGD_optimizer.step()
                else:
                    updated_model, _ = self.SGD_optimizer.step(init_model.parameters())
                    
        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)


        # self.update_parameters(updated_model)

        return LOSS

    def train_prox(self, epochs,mu_t, gen_ws=None):
        LOSS = 0
        self.model.train()

        for epoch in range(1, epochs+ 1):  # local update
            self.model.train()
            # batch_idx=0
            for X, y in self.trainloader:
                # print(f"batch index{batch_idx}:")
                # batch_idx+= 1
                X, y = X.to(self.device), y.to(self.device)  # self.get_next_train_batch()
                self.SGD_optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)

                loss.backward()
                updated_model, _ = self.SGD_optimizer.step(mu_t,gen_ws)
                # if (gen_ws is None):
                #     # updated_model, _ = self.SGD_optimizer.step()
                #     _ = self.SGD_optimizer.step()
                # else:
                #     updated_model, _ = self.SGD_optimizer.step(gen_ws)

        # update local model as local_weight_upated
        # self.clone_model_paramenter(self.local_weight_updated, self.local_model)

        # self.update_parameters(updated_model)

        return LOSS

    def train_distill(self, epochs,mu_t, gen_model=None):
        LOSS = 0
        self.model.train()
        # gen_model.train()

        for epoch in range(1, epochs + 1):  # local update
            self.model.train()
            # gen_model.train()
            # batch_idx=0
            for X, y in self.trainloader:
                # print(f"batch index{batch_idx}:")
                # batch_idx+= 1
                X, y = X.to(self.device), y.to(self.device)  # self.get_next_train_batch()
                self.SGD_optimizer.zero_grad()
                output = self.model(X)
                gen_output = gen_model(X)

                lossTrue = self.loss(output, y)
                lossKD= self.criterion_KL(output,gen_output)
                loss = lossTrue + 1* lossKD
                loss.backward()

                updated_model, _ = self.SGD_optimizer.step(mu_t=0, gen_weights=None)
        return LOSS

    def train_prox_distill(self, epochs,mu_t, gen_model,distill_model=None):
        LOSS = 0
        self.model.train()
        # gen_model.train()

        for epoch in range(1, epochs + 1):  # local update
            self.model.train()
            # gen_model.train()
            # batch_idx=0
            for X, y in self.trainloader:
                # print(f"batch index{batch_idx}:")
                # batch_idx+= 1
                X, y = X.to(self.device), y.to(self.device)  # self.get_next_train_batch()
                self.SGD_optimizer.zero_grad()
                output = self.model(X)
                if (distill_model == None):
                    distill_model = gen_model
                    lossKD1 = 0

                # gen_output = distill_model(X)
                gen_output1 = gen_model(X)
                lossKD1 = self.criterion_KL(output, gen_output1)

                lossTrue = self.loss(output, y)
                # lossKD= self.criterion_KL(output,gen_output)

                # loss = lossTrue + 1* lossKD + 1* lossKD1
                loss = lossTrue + 1 * lossKD1
                loss.backward()

                updated_model, _ = self.SGD_optimizer.step(mu_t=mu_t, gen_weights=(gen_model,1.0))
        return LOSS
