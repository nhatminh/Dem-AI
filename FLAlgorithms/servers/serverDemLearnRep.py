import copy

import torch
import os
import torch.multiprocessing as mp
from tqdm import tqdm

from FLAlgorithms.servers.serverbase_dem import Dem_Server
from FLAlgorithms.users.userDemLearn import UserDemLearn
# from FLAlgorithms.servers.serverbase import Server
# from Setting import *
from utils.data_utils import write_file
from utils.dem_plot import *
from utils.model_utils import *
import numpy as np

# Implementation for pFedMe Server

class DemLearnRep(Dem_Server):
    def __init__(self, experiment, device, dataset, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, cutoff, args):
        # if (args.mu > 0): args.algorithm += "_Prox"

        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times, args)

        # Initialize data for all  users
        self.K = K
        self.personal_learning_rate = personal_learning_rate

        # total_users = len(dataset[0][0])
        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(self.total_users)

        for i in range(self.total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            print("User ", id, ": Numb of Training data", len(train))
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)
                    
            user = UserDemLearn(device, id, train, test, model, batch_size, learning_rate, beta, L_k, local_epochs, optimizer, K, personal_learning_rate, args)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Fraction number of users / total users:",num_users, " / " ,self.total_users)
        print(f"Finished creating {self.args.algorithm} server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        mu_t = self.args.mu
        for i in range(self.num_glob_iters):
            # ============= Test each client =============
            tqdm.write('============= Test Client Models - Specialization ============= ')
            stest_acu, strain_acc = self.evaluating_clients(i, mode="spe")
            self.cs_avg_data_test.append(stest_acu)
            self.cs_avg_data_train.append(strain_acc)
            tqdm.write('============= Test Client Models - Generalization ============= ')
            gtest_acu, gtrain_acc = self.evaluating_clients(i, mode="gen")
            self.cg_avg_data_test.append(gtest_acu)
            self.cg_avg_data_train.append(gtrain_acc)

            # ============= Test root =============
            if (i > 0):
                tqdm.write('============= Test Group Models - Specialization ============= ')
                self.evaluating_groups(self.TreeRoot, i, mode="spe")
                # gs_test = self.test_accs / self.count_grs
                # gs_train = self.train_accs / self.count_grs
                # self.gs_data_test.append(gs_test)
                # self.gs_data_train.append(gs_train)
                self.gs_level_train[:, i, 0] = self.gs_level_train[:, i, 0] / self.gs_level_train[:, i,
                                                                              1]  # averaging by level and numb of clients
                self.gs_level_test[:, i, 0] = self.gs_level_test[:, i, 0] / self.gs_level_test[:, i,
                                                                            1]  # averaging by level and numb of clients
                print("AvgG. Testing performance for each level:", self.gs_level_test[:, i, 0])
                # print("AvgG. Training performance for each level:", self.gs_level_train[:,i,0])
                tqdm.write('============= Test Group Models - Generalization ============= ')
                self.evaluating_groups(self.TreeRoot, i, mode="gen")
                # gg_test = self.test_accs / self.count_grs
                # gg_train = self.train_accs / self.count_grs
                # self.gg_data_test.append(gg_test)
                # self.gg_data_train.append(gg_train)
                self.gg_level_train[:, i, 0] = self.gg_level_train[:, i, 0] / self.gg_level_train[:, i,
                                                                              1]  # averaging by level and numb of clients
                self.gg_level_test[:, i, 0] = self.gg_level_test[:, i, 0] / self.gg_level_test[:, i,
                                                                            1]  # averaging by level and numb of clients
                print("AvgG. Testing performance for each level:", self.gg_level_test[:, i, 0])
                # print("AvgG. Training performance for each level:", self.gg_level_train[:,i,0])


            if(self.experiment):
                self.experiment.set_epoch( i + 1)
            print("-------------Round number: ",i, " -------------")
            # # send all parameter for users
            # self.send_parameters()

            # # Evaluate gloal model on user for each iteration
            # print("Evaluate global model")
            # print("")
            # self.evaluate()

            # do update for all users not only selected users
            #for user in self.users:
            #    user.train(self.local_epochs) #* user.train_sample

            for user in self.users:
                # STEP 1: Initialize the local model of client based on hierarchical GK
                if (i == 0):
                    self.initialize_model(user)
                    user.train_prox(self.local_epochs, mu_t=self.args.mu)
                else:
                    # parent_weights = user.model
                    ### Best Performance near FedAvg ###
                    parent_weights = user.get_parent_model() #super group representation
                    # self.initialize_model(user, parent_weights) # we better to start with this group model to avoid overfitting problem

                    if(i<21): #Focus on Generalization: DemLearn
                        user.train_prox(self.local_epochs, mu_t=mu_t, gen_ws=(parent_weights, 1.))

                    else: #Focus on Specialization: DemLearnRep
                        c_weights = user.model.parameters()
                        w_cnt = 0
                        #Head Update
                        head_idx = (self.args.K_Layer_idx[0]-1) * 2 ## 3*2 = 6 (head start from element 6th)

                        for c_w, g_w in zip(c_weights, parent_weights.parameters()):
                            if (w_cnt < head_idx):  # load generalized representation
                                # c_w.data = ((1 - self.beta) * (c_w.data) + self.beta * g_w.data)
                                c_w.data = g_w.data
                            else: # do not need to load head layers
                                break
                            w_cnt += 1

                        # STEP 2: Local training to update local representation and head
                        #### Update Local Heads tau1-steps ####
                        pre_w = copy.deepcopy(user.model)
                        # print("Local Training at user ",user.id)
                        # user.train(self.local_epochs,mu_t,mode="head")
                        user.train_prox(1, mu_t=0, gen_ws=(parent_weights, 1.))

                        new_w = user.model
                        self.copy_rep(pre_w, new_w)  #fix rep

                        #### Update Local Reprensetation tau2-step ####
                        for n in range(self.local_epochs):
                            pre_w = copy.deepcopy(user.model)
                            # user.train(1,mu_t,mode="rep")
                            user.train_prox(1, mu_t=mu_t, gen_ws=(parent_weights, 1.))
                            self.copy_rep(new_w, pre_w) #fix head
                            user.set_parameters(pre_w)

            # if (self.args.DECAY == True):
            #     if(DATASET=="mnist"):
            #         print("DECAYING Beta:",self.beta)
            #         self.beta = max(self.beta *0.7, 0.001) #### Mnist dataset
            #     elif(DATASET == "fmnist"):
            #         self.beta = max(self.beta * 0.5, 0.0005)  #### Fmnist dataset
            #     elif(DATASET == "femnist"):
            #         self.beta = max(self.beta * 0.5, 0.0005)  #### FEmnist dataset
            #     # self.gamma = max(self.gamma - 0.25, 0.02)  # period = 2  0.96 vs 0.9437 after 31 : 0.25, 0.02 DemAVG
            #     # self.gamma = max(self.gamma - 0.1, 0.6) # 0.25, 0.02:  0.987 vs 0.859 after 31 DemProx vs fixed 0.6 =>0.985 and 0.89

            # STEP 3: Hierarchical Clustering and Averaging
            if (i % TREE_UPDATE_PERIOD == 0):
                print("DEM-AI --------->>>>> Clustering")
                self.hierrachical_clustering(i,rep=True)
                # self.TreeRoot.print_structure()
                print("DEM-AI --------->>>>> Hard Update generalized model")
                # self.update_generalized_model(self.TreeRoot)  # hard update
                self.update_generalized_model_recursive2_rep(i, self.TreeRoot)
                # print("Root Model:", np.sum(self.TreeRoot.gmodel[0]),np.sum(self.TreeRoot.gmodel[1]))
            else:
                # update model
                # self.latest_model = self.aggregate(csolns,weighted=True)
                print("DEM-AI --------->>>>> Soft Update generalized model")
                # self.update_generalized_model(self.TreeRoot, mode="soft")  # soft update
                self.update_generalized_model_recursive2_rep(i, self.TreeRoot)
                # print("Root Model:", np.sum(self.TreeRoot.gmodel[0]),np.sum(self.TreeRoot.gmodel[1]))


            # # choose several users to send back upated model to server
            # # self.personalized_evaluate()
            # self.selected_users = self.select_users(i,self.num_users)

            # # Evaluate gloal model on user for each interation
            # #print("Evaluate persionalized model")
            # #print("")
            # self.evaluate_personalized_model()
            # #self.aggregate_parameters()
            # self.persionalized_aggregate_parameters()
            
        # self.save_results()
        # self.save_model()
        self.save_results_dem()

    def save_results_dem(self):

        # plt.plot(np.arange(1,len(self.time_complex)+1),self.time_complex)
        # plt.show()
        root_train = np.asarray(self.gs_level_train)[K_Levels, :, 0]
        root_test = np.asarray(self.gs_level_test)[K_Levels, :, 0]

        # write_file(RS_PATH+self.args.algorithm+"_"+ complex_file_path, time_complex=self.time_complex)
        # print('Saving complexity..\nOK!')
        rs_path = RS_PATH + self.args.algorithm + "_" + rs_file_path
        write_file(file_name=rs_path, root_test=root_test, root_train=root_train,
                   cs_avg_data_test=self.cs_avg_data_test, cs_avg_data_train=self.cs_avg_data_train,
                   cg_avg_data_test=self.cg_avg_data_test, cg_avg_data_train=self.cg_avg_data_train,
                   cs_data_test=self.cs_data_test, cs_data_train=self.cs_data_train, cg_data_test=self.cg_data_test,
                   cg_data_train=self.cg_data_train, gs_level_train=self.gs_level_train,
                   gs_level_test=self.gs_level_test,
                   gg_level_train=self.gg_level_train, gg_level_test=self.gg_level_test,
                   gks_level_train=self.gks_level_train, gks_level_test=self.gks_level_test,
                   gkg_level_train=self.gkg_level_train, gkg_level_test=self.gkg_level_test,
                   dendo_data=self.dendo_data, dendo_data_round=self.dendo_data_round,  # Dendrogram data
                   N_clients=[N_clients], TREE_UPDATE_PERIOD=[TREE_UPDATE_PERIOD])  # Setting
        plot_from_file(rs_path)
    
  
