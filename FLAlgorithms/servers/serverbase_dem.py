import timeit

import torch
import os
import numpy as np
import h5py
from tqdm import tqdm

from FLAlgorithms.servers.serverbase import Server
from utils.clustering.hierrachical_clustering import tree_construction, cal_linkage_matrix, weight_clustering, \
    cosine_clustering
from utils.model_utils import Metrics
import copy
from Setting import *

class Dem_Server(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate ,beta, L_k,
                 num_glob_iters, local_epochs, optimizer,num_users, times, args=None):
        super().__init__(experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        #DEMLEARN PARAMS

        if(args is not None):

            self.alpha_factor = args.alpha_factor
            if (args.mu > 0 and "dem" in args.algorithm):
                args.algorithm += "_Prox"

            self.algorithm = args.algorithm
            self.num_rounds = num_glob_iters
            self.total_users = args.total_users
            self.N_clients = self.total_users
            self.args = args
            self.count_grs = np.zeros(K_Levels + 1)

        self.cg_avg_data_test = []  # avg generalization client accuracy test
        self.cg_avg_data_train = []  # avg generalization client accuracy train
        self.cs_avg_data_test = []  # avg specialization client test accuracy
        self.cs_avg_data_train = []  # avg specialization client train accuracy
        # self.gs_data_test = []  # specialization of group test accuracy
        # self.gs_data_test.append(np.zeros(K_Levels+1 ))
        # self.gg_data_test = [] # generalization of group test accuracy
        # self.gg_data_train = []  # generalization of group train accuracy
        # self.gs_data_train = []  # specialization of group train accuracy
        # self.gs_data_train.append(np.zeros(K_Levels + 1))

        self.cs_data_test = np.zeros((self.num_rounds, self.N_clients))
        self.cs_data_train = np.zeros((self.num_rounds, self.N_clients))
        self.cg_data_test = np.zeros((self.num_rounds, self.N_clients))
        self.cg_data_train = np.zeros((self.num_rounds, self.N_clients))
        self.gs_level_train = np.zeros((K_Levels + 1, self.num_rounds, 2))  # specialization of group train accuracy
        self.gs_level_test = np.zeros((K_Levels + 1, self.num_rounds, 2))  # specialization of group test accuracy
        self.gks_level_train = np.zeros((2, self.num_rounds))  # specialization of group k train accuracy
        self.gks_level_test = np.zeros((2, self.num_rounds))  # specialization of group k test accuracy
        self.gg_level_train = np.zeros((K_Levels + 1, self.num_rounds, 2))  # generalization of group train accuracy
        self.gg_level_test = np.zeros((K_Levels + 1, self.num_rounds, 2))  # generalization of group test accuracy
        self.gkg_level_train = np.zeros((2, self.num_rounds))  # generalization of group k train accuracy
        self.gkg_level_test = np.zeros((2, self.num_rounds))  # generalization of group k test accuracy
        self.dendo_data = []
        self.dendo_data_round = []
        self.time_complex = []

    def run_clustering_rep(self):
        # print("--Rep Clustering")
        p_list = []
        for c in self.users:
            # print("Weight:", w[1][0])
            # print("Bias:", w[1][1])
            # weight_idx = (K_Layer_idx[0] - 1)*2 #the layer for clustering algorithm e.g.: (3-1) *2 =4 (layer3, index 4, 5)
            weight_idx = (K_Layer_idx[1] - 1) * 2  # the layer for clustering algorithm e.g.: (3-1) *2 =4 (layer3, index 4, 5) (we can choose 6, 7 as layer 4 to cluster)
            w_cnt = 0
            # if (CLUSTER_METHOD == "weight"):
            c_w = []
            # for w in range(2):  #using the first layer to cluster only.
            c_model = copy.deepcopy(c.model)
            for w in c_model.parameters():
                # print("layer element idx:",w_cnt)
                if (w_cnt < weight_idx+2): # using all layers of rep for clustering (i.e., layer 1,2,3)
                # if (w_cnt > weight_idx) and (w_cnt < weight_idx + 2):  # using one layer for clustering (i.e., layer 3 => last layer of rep)
                    c_w.append(w.data.flatten().cpu().numpy())
                w_cnt += 1

            # print("Concatenation Len for clustering:",len(np.concatenate( c_w, axis=0)))
            p_list.append(np.concatenate(c_w, axis=0))

            # else:
            #     c_g = []
            #     # for g in range(2): #using the first layer to cluster only.
            #     for g in c.model.parameters():
            #         if (w_cnt > weight_idx) and (
            #                 w_cnt < weight_idx + 2):  # using one layer for clustering (i.e., layer 3 => last layer of rep)
            #             c_g.append(g.data.flatten().cpu().numpy())
            #         w_cnt += 1
            #     p_list.append(np.concatenate(c_g, axis=0))

        self.Weight_dimension = len(p_list[0])
        if (CLUSTER_METHOD == "weight"):
            return weight_clustering(p_list)
        else:
            return cosine_clustering(p_list)

    def run_clustering(self):
        p_list = []
        for c in self.users:
            # print("Weight:", w[1][0])
            # print("Bias:", w[1][1])
            # if (CLUSTER_METHOD == "weight"):
            c_w = []
            w_cnt = 0
            # for w in range(2):  #using the first layer to cluster only.
            c_model = copy.deepcopy(c.model)
            for w in c_model.parameters():
                # if(w_cnt<2):
                c_w.append(w.data.flatten().cpu().numpy())
                w_cnt +=1
            # print("Concatenation Len for clustering:",len(np.concatenate( c_w, axis=0)))
            # print(f"Numb of parameters elements in NN {w_cnt}")
            concat_w = np.concatenate(c_w, axis=0)
            # print(len(concat_w))
            p_list.append(concat_w)

            # else:
            #     c_g = []
            #     # for g in range(2): #using the first layer to cluster only.
            #     for g in c.model.parameters():
            #         c_g.append(g.data.flatten().cpu().numpy())
            #     p_list.append(np.concatenate(c_g, axis=0))

        self.Weight_dimension = len(p_list[0])
        if (CLUSTER_METHOD == "weight"):
            return weight_clustering(p_list)
        else:
            return cosine_clustering(p_list)

    def initialize_model(self, user, init_model=None):
        assert (self.users is not None and len(self.users) > 0)
        if(init_model == None):
            user.set_parameters(self.model)
        else:
            user.set_parameters(init_model)

    # copy rep of left to right:
    def copy_rep(self, pre_w, new_w):
        w_cnt = 0
        head_idx = (self.args.K_Layer_idx[0]-1) * 2 ## 3*2 = 6 (head start from element 6th)
        for p_w, n_w in zip(pre_w.parameters(), new_w.parameters()):
            if (w_cnt < head_idx):  # load representation
                n_w.data = p_w.data
            else:  # do not need to load head layers
                break
            w_cnt += 1

    # copy rep of left to right:
    def copy_sub_rep(self, pre_w, new_w,rep_start,rep_end):
        w_cnt = 0
        for p_w, n_w in zip(pre_w.parameters(), new_w.parameters()):
            if (w_cnt >= rep_start) and (w_cnt <= rep_end):  # load representation
                n_w.data = p_w.data
            w_cnt += 1

    # # copy new presentation to replace old representation => fix head
    # def reverse_head(self, pre_w, new_w):
    #     w_cnt = 0
    #     head_idx = (self.args.K_Layer_idx[0]-1) * 2 ## 3*2 = 6 (head start from element 6th)
    #     for p_w, n_w in zip(pre_w.parameters(), new_w.parameters()):
    #         if (w_cnt < head_idx):  # load representation
    #             p_w.data = n_w.data
    #         else:  # do not need to load head layers
    #             break
    #         w_cnt += 1

    def update_generalized_model(self, round, node, mode="hard"):
        # print("Node id:", node._id, node._type)
        childs = node.childs
        if childs:
            # node.numb_clients = node.count_clients()
            node.in_clients = node.collect_clients()
            node.numb_clients = len(node.in_clients)
            # print(node.numb_clients)
            # print(self.Weight_dimension)

            rs_ws = copy.deepcopy(self.update_generalized_model(round,childs[0], mode))
            for rs_w in rs_ws.parameters():
                rs_w.data = rs_w.data *childs[0].numb_clients
            for child in childs[1:]:
                child_weights = self.update_generalized_model(round,child, mode)
                for rs_w, child_w in zip(rs_ws.parameters(), child_weights.parameters()):
                    rs_w.data = rs_w.data + child.numb_clients * child_w.data  # weight

            if(mode == "hard"):
                for rs_w in rs_ws.parameters():
                    rs_w.data = rs_w.data / node.numb_clients
                    # if node.level == 1 and round<10:
                    if round < 5 and Amplify:
                        if round > 2 and DATASET == "Cifar10":
                            rs_w.data = rs_w.data
                        else:
                            rs_w.data *= 1.15

            elif(mode == "soft"):
                for o_w, rs_w in zip( node.gmodel.parameters(),rs_ws.parameters()):
                    rs_w.data = (1 - self.args.gamma) * o_w.data + self.args.gamma * rs_w.data / node.numb_clients

            # node.gmodel = rs_ws
            node.set_node_parameters(rs_ws)
            return rs_ws

        elif (node._type.upper() == "CLIENT"):  # Client
            return node.model

    def update_generalized_model_re(self, round, node, mode="hard"):
        # print("Node id:", node._id, node._type)
        childs = node.childs
        if childs:
            # node.numb_clients = node.count_clients()
            node.in_clients = node.collect_clients()
            node.numb_clients = len(node.in_clients)
            # print(node.numb_clients)
            # print(self.Weight_dimension)
            rs_ws = copy.deepcopy(self.update_generalized_model_re(round,childs[0], mode))
            for rs_w in rs_ws.parameters():
                rs_w.data = rs_w.data *childs[0].numb_clients
            for child in childs[1:]:
                child_weights = self.update_generalized_model_re(round,child, mode)
                for rs_w, child_w in zip(rs_ws.parameters(), child_weights.parameters()):
                    rs_w.data = rs_w.data + child.numb_clients * child_w.data  # weight

            if(node.parent=="Empty"):
                for rs_w in rs_ws.parameters():
                    rs_w.data = rs_w.data / node.numb_clients
            else:
                for p_w, rs_w in zip( node.parent.gmodel.parameters(),rs_ws.parameters()):
                    rs_w.data = (p_w.data + rs_w.data / node.numb_clients) *0.5

            # node.gmodel = rs_ws
            node.set_node_parameters(rs_ws)
            return rs_ws

        elif (node._type.upper() == "CLIENT"):  # Client
            return node.model

    def update_generalized_model_downward(self, round, node, mode="hard",rep=False):
        # print("Node id:", node._id, node._type)

        childs = node.childs
        if childs:
            node_params = node.gmodel
            for child in childs[0:]:
                # print(f"child index {child._id} level {child.level}")
                if(child.level > 0):
                    if(rep and (round > 20)):
                    # if(False):
                        w_cnt = 0
                        spe_rep_idx = (self.args.K_Layer_idx[child.level]-1) * 2  ## 2*2 = 2 (group_spe_rep start from element 4th)
                        print("rep idx: ",spe_rep_idx)
                        for c_w, n_w in zip(child.gmodel.parameters(), node_params.parameters()):
                            if (w_cnt < spe_rep_idx):  # load generalized representation
                                c_w.data = (1 - self.alpha_factor) * c_w.data + n_w.data * self.alpha_factor
                            else:  # do not need to load head layers
                                break
                            w_cnt += 1
                    else:
                        for c_w, n_w in zip(child.gmodel.parameters(), node_params.parameters()):
                            c_w.data = (1 - self.alpha_factor) * c_w.data + n_w.data * self.alpha_factor

                    self.update_generalized_model_downward(round,child,mode,rep)
                else:  ## We used this sum of local and group model to get final local model in each round
                    if (rep and (round >20)):  # update rep of child model only
                        w_cnt = 0
                        head_idx = (self.args.K_Layer_idx[0] - 1) * 2  ## 3*2 = 6 (head start from element 6th)
                        for c_w, n_w in zip(child.model.parameters(), node_params.parameters()):
                            if (w_cnt < head_idx):  # load generalized representation
                                c_w.data = (1 - self.alpha_factor) * c_w.data + n_w.data * self.alpha_factor
                            else:  # do not need to load head layers
                                break
                            w_cnt += 1

                    else:  # update full child model
                        for c_w, n_w in zip(child.model.parameters(), node_params.parameters()):
                            c_w.data = (1 - self.alpha_factor) * c_w.data + n_w.data * self.alpha_factor
                            # c_w.data = c_w.data


    def update_generalized_model_recursive2(self, round, node, mode="hard"):
        # mu_factor=0.6 #0.5 is the best value
        print("self.alpha_factor:", self.alpha_factor)
        if self.args.alpha_factor ==0:
          self.alpha_factor= 0.7
          self.alpha_factor = max(self.alpha_factor/1.02, 0.6)

        # self.mu_factor=0.6
        self.update_generalized_model(round,node,mode) #bottom to top
        self.update_generalized_model_downward(round,node,mode) #top to bottom

    def update_generalized_model_recursive2_rep(self, round, node, mode="hard"):
        # mu_factor=0.6 #0.5 is the best value
        print("self.alpha_factor:", self.alpha_factor)
        if self.args.alpha_factor ==0:
          self.alpha_factor= 0.7
          self.alpha_factor = max(self.alpha_factor/1.02, 0.6)

        self.update_generalized_model(round,node,mode) #bottom to top
        self.update_generalized_model_downward(round,node,mode,rep=True) #top to bottom


    def update_generalized_model_recursive1(self, round, node, mode="hard"):
        if(mode=="hard"):
            self.update_generalized_model_recursive2(round,node, mode)
        else:
            self.update_generalized_model_re(round,node,mode) #bottom to top in recursive way


    def get_hierrachical_params(self, client):
        gen_weights, nf = client.get_hierarchical_info1()
        # print("Normalized term:", nf)
        for w in gen_weights.parameters():
            w.data = w.data /nf

        # total_corrects, ns = self.gc_test(gen_weights)
        # print(f"G_GEN from download model: {total_corrects / ns}")
        return gen_weights.parameters() # normalized version

    def get_hierrachical_gen_model(self, client):
        # gen_weights, nf = client.get_hierarchical_info1()
        gen_weights, nf = client.get_hierarchical_info()

        for w in gen_weights.parameters():
            w.data = w.data/nf
        return gen_weights, 1.0

        # return client.get_hierrachical_gen_model()
        # return client.get_hierarchical_info1()
        # return client.get_hierarchical_info()

    def get_hierrachical_representation(self, client):
        return client.get_hierarchical_rep().parameters()

        ## model shape type: tuple ((weight, bias),(weight, bias))


    def hierrachical_clustering(self, i,rep=False):
        # if(self.Hierrchical_Method == "Weight"):
        #     weights_matrix = self.create_matrix()
        #
        # else:
        #     gradient_matrix = self.create_matrix()
        #     # gradient_matrix = np.random.rand(N_clients, Weight_dimension)
        #     model = gradient_clustering(gradient_matrix)

        # start_cluster = timeit.timeit()
        if(self.args.algorithm == "DemLearn" or self.args.algorithm == "DemLearn_Prox"):
            model = self.run_clustering()
        elif(self.args.algorithm == "DemLearnRep" or self.args.algorithm == "DemLearnRep_Prox"):
            if (rep and (i >20)):
                model = self.run_clustering_rep()
            else:
                model = self.run_clustering()
            # model = self.run_clustering()
        # end_cluster = timeit.timeit()
        # total_time = start_cluster - end_cluster
        # self.time_complex.append(total_time)

        self.TreeRoot = tree_construction(model, self.users, round=i)
        # self.dendo_data.append([model, i])
        rs_linkage = cal_linkage_matrix(model)[1]
        self.dendo_data.append(rs_linkage)
        self.dendo_data_round.append(i)
        # self.dendo_data.append([rs_linkage, i])
        # plot_dendrogram(rs_linkage, round, self.alg)

        print("Number of agents in tree:", self.TreeRoot.count_clients())
        print("Number of agents in level K:", self.TreeRoot.childs[0].count_clients(),
              self.TreeRoot.childs[1].count_clients())
        # print("Number of agents Group 1 in level K-1:", root.childs[0].childs[0].count_clients(),
        #       root.childs[0].childs[1].count_clients())

    def g_train_error_and_loss(self, gr, mode="spe"):  # mode spe: specialization, gen: generalization
        num_samples = []
        tot_correct = []
        losses = []

        self.client_model.set_params(gr.gmodel)  # update parameter of group to tf.graph
        if (mode == "spe"):
            validating_clients = gr.in_clients
        elif (mode == "gen"):
            validating_clients = self.users

        for c in validating_clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]

        return ids, groups, num_samples, tot_correct, losses

    # C-GEN for Training data
    def gc_train_error_and_loss(self):
        num_samples = []
        tot_correct = []

        for c in self.users:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

        return np.sum(tot_correct), np.sum(tot_correct), np.sum(num_samples)

    def c_train_error_and_loss(self, i, mode="spe"):  # mode spe: specialization, gen: generalization
        num_samples = []
        tot_correct = []
        losses = []
        clients_acc = []

        for c in self.users:
            if (mode == "spe"):
                ct, cl, ns = c.train_error_and_loss()
            elif (mode == "gen"):
                ct, cl, ns = self.gc_train_error_and_loss()  # Test client as testing group approach in gen mode

            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            clients_acc.append(ct / ns)

        # print("Training Acc Client:", clients_acc)
        # self.client_data_train[i][:] = clients_acc

        if (mode == "spe"):
            self.cs_data_train[i, :] = clients_acc
        elif (mode == "gen"):
            self.cg_data_train[i, :] = clients_acc

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]

        return ids, groups, num_samples, tot_correct, losses

    def g_test(self, gr, mode="spe"):  # mode spe: specialization, gen: generalization
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []

        # print("Clients in group:",self.gr.in_clients)
        if (mode == "spe"):
            validating_clients = gr.in_clients  # Evaluate Group-SPE (clients belong to this group)
        elif (mode == "gen"):
            validating_clients = self.users     # Evaluate Group-GEN (all clients data)

        for c in validating_clients:
            # gr_test_model = copy.deepcopy(gr.gmodel)
            # gr_test_model= gr.gmodel.clone()
            # ct, ns = c.test_gen(gr_test_model)
            ct, ns = c.test_gen(gr.gmodel)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]

        return ids, groups, num_samples, tot_correct

    # C-GEN for Testing data
    def gc_test(self,c_model):
        num_samples = []
        tot_correct = []

        for c in self.users:
            ct, ns = c.test_gen(c_model)
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)

        return np.sum(tot_correct), np.sum(num_samples)

    def c_test(self, i, mode="spe"):  # mode spe: specialization, gen: generalization
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        clients_acc = []

        for c in self.users:
            if (mode == "spe"):
                ct, ns, _ = c.test()
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
            elif (mode == "gen"):
                # c_test_model = copy.deepcopy(c.model)
                # ct, ns = self.gc_test(c_test_model)  # Test client as testing group approach in gen mode
                ct, ns = self.gc_test(c.model)  # Test client as testing group approach in gen mode

            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            clients_acc.append(ct / ns)

        # print("Testing Acc Client:", clients_acc )
        if (mode == "spe"):
            self.cs_data_test[i, :] = clients_acc
        elif (mode == "gen"):
            self.cg_data_test[i, :] = clients_acc
        # self.client_data_test.append(clients_acc)

        ids = [c.id for c in self.users]
        groups = [c.group for c in self.users]
        return ids, groups, num_samples, tot_correct

    def evaluating_clients(self, i, mode="spe"):  # mode spe: specialization, gen: generalization
        stats = self.c_test(i, mode)  #evaluate C-GEN and C-SPE on testing data
        test_acr = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
        tqdm.write('At round {} AvgC. testing accuracy: {}'.format(i, test_acr))

        train_acr=[]
        # if mode == "spe":
        #     stats_train = self.c_train_error_and_loss(i, mode) #evaluate C-SPE on training data
        # elif (mode == "gen"):
        #     stats_train = []  ### no need to evaluate C-GEN on training data
        # if mode == "spe":
        #     train_acr = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
        #     tqdm.write('At round {} AvgC. training accuracy: {}'.format(i, train_acr))
        # elif (mode == "gen"):
        #     train_acr = []
        return test_acr, train_acr

    def evaluating_groups(self, gr, i, mode="spe"):  # mode spe: specialization, gen: generalization
        if (gr.parent == "Empty"):
            self.test_accs = np.zeros(K_Levels + 1)
            self.train_accs = np.zeros(K_Levels + 1)
            self.count_grs = np.zeros(K_Levels + 1)

        stats = self.g_test(gr, mode)
        test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])

        if (mode == "spe"):  # Specialization results
            self.gs_level_train[gr.level - 1, i, 1] += gr.numb_clients
            self.gs_level_test[gr.level - 1, i, 0] += test_acc * gr.numb_clients
            self.gs_level_test[gr.level - 1, i, 1] += gr.numb_clients

            if (gr._id == self.TreeRoot.childs[0]._id):
                # self.gks_level_train[0,i] = train_acc
                self.gks_level_test[0, i] = test_acc
            elif (gr._id == self.TreeRoot.childs[1]._id):
                # self.gks_level_train[1,i] = train_acc
                self.gks_level_test[1, i] = test_acc

        elif (mode == "gen"):  # Generalization results
            # self.gg_level_train[gr.level - 1, i, 0] += train_acc * gr.numb_clients
            self.gg_level_train[gr.level - 1, i, 1] += gr.numb_clients
            self.gg_level_test[gr.level - 1, i, 0] += test_acc * gr.numb_clients
            self.gg_level_test[gr.level - 1, i, 1] += gr.numb_clients
            if (gr._id == self.TreeRoot.childs[0]._id):
                # self.gkg_level_train[0,i] = train_acc
                self.gkg_level_test[0, i] = test_acc
            elif (gr._id == self.TreeRoot.childs[1]._id):
                # self.gkg_level_train[1,i] = train_acc
                self.gkg_level_test[1, i] = test_acc

        if (gr.childs):
            for c in gr.childs:
                if (c._type.upper() == "GROUP"):
                    self.evaluating_groups(c, i, mode)