import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer, Prox_SGD, DemSGD, DemProx_SGD
from FLAlgorithms.users.userbase_dem import User
import copy


# Implementation for pFeMe clients

class UserDemLearn(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k,
                 local_epochs, optimizer, K, personal_learning_rate, args):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, L_k,
                         local_epochs)

        if (model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.args = args
        self.mu = args.mu

        # self.SGD_optimizer = DemSGD(self.model.parameters(), lr=self.personal_learning_rate)

        # if(self.mu==0):
        #     print("Using DemSGD Optimizer")
        #     self.Prox_optimizer = DemSGD(self.model.parameters(), lr=self.personal_learning_rate)
        # else:
        #     print("Using ProxSGD Optimizer")
        #     self.Prox_optimizer = Prox_SGD(self.model.parameters(), lr=self.personal_learning_rate, mu=self.mu)
        # if (self.mu == 0):
        #     print("Using DemSGD Optimizer")
        #     # self.SGD_optimizer = DemSGD(self.model.parameters(), lr=self.personal_learning_rate)
        #     self.SGD_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # else:
        print("Using ProxSGD Optimizer")
        self.SGD_optimizer = DemProx_SGD(self.model.parameters(), lr=self.learning_rate, mu=self.mu)
        self.Prox_optimizer = DemProx_SGD(self.model.parameters(), lr=self.learning_rate, mu=self.mu)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs, mu_t, mode="head"):
        LOSS = 0
        # self.model.train()
        if (self.mu > 0):
            init_model = copy.deepcopy(self.model)

        for epoch in range(1, epochs + 1):  # local update
            self.model.train()
            for X, y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)  # self.get_next_train_batch()
                # X, y = self.get_next_train_batch()
                # K = 30 # K is number of personalized steps
                if (mode == "head"):
                    self.SGD_optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.loss(output, y)
                    loss.backward()
                    updated_model, _ = self.SGD_optimizer.step()
                else:
                    self.Prox_optimizer.zero_grad()
                    output = self.model(X)
                    loss = self.loss(output, y)
                    loss.backward()
                    if (self.mu == 0):
                        updated_model, _ = self.Prox_optimizer.step()
                    else:
                        updated_model, _ = self.Prox_optimizer.step(init_model.parameters())

                # # update local weight after finding aproximate theta
                # for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                #     localweight.data = localweight.data - self.L_k* self.learning_rate * (localweight.data - new_param.data)

        # update local model as local_weight_upated
        # self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(updated_model)

        return LOSS

    def train_prox(self, epochs, mu_t, gen_ws=None):
        LOSS = 0
        self.model.train()

        for epoch in range(1, epochs + 1):  # local update
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
                if(mu_t ==0):
                    updated_model, _ = self.SGD_optimizer.step(mu_t, gen_ws)
                else:
                    updated_model, _ = self.Prox_optimizer.step(mu_t, gen_ws)

                # if (gen_ws is None):
                #     # updated_model, _ = self.SGD_optimizer.step()
                #     _ = self.SGD_optimizer.step()
                # else:
                #     updated_model, _ = self.SGD_optimizer.step(gen_ws)

        # update local model as local_weight_upated
        # self.clone_model_paramenter(self.local_weight_updated, self.local_model)

        # self.update_parameters(updated_model)

        return LOSS
