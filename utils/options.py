#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
from Setting import *

def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser = argparse.ArgumentParser()
    if DATASET == "mnist": DefaultDataset = "Mnist"
    else: DefaultDataset = DATASET

    if RUNNING_ALG == "demlearn-p": DefaultALG="DemLearn"
    else: DefaultALG = RUNNING_ALG

    parser.add_argument("--dataset", type=str, default=DefaultDataset, choices=["fmnist","femnist","human_activity", "gleam","vehicle_sensor","mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=Learning_rate, help="Local learning rate")
    parser.add_argument("--L_k", type=float, default=1, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=NUM_GLOBAL_ITERS)
    parser.add_argument("--local_epochs", type=int, default = LOCAL_EPOCH)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default=DefaultALG,choices=["pFedMe", "pFedMe_p", "PerAvg", "fedprox","FedAvg", "FedU", "Mocha", "Local" , "Global","DemLearn","DemLearnRep"])
    parser.add_argument("--subusers", type = float, default = 1, help="Number of Users per round")  #Fraction number of users
    parser.add_argument("--K", type=int, default=0, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.02, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to commet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments") #GPU dev_id, -1 is CPU
    parser.add_argument("--cutoff", type=int, default=0, help="Cutoff data sample")
    parser.add_argument("--beta", type=float, default=PARAMS_beta, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--DECAY", type=bool, default=DECAY, help="DECAY or CONSTANT")
    parser.add_argument("--mu", type=int, default=PARAMS_mu, help="mu parameter")
    parser.add_argument("--gamma", type=int, default=PARAMS_gamma, help="gama parameter")
    parser.add_argument("--total_users", type=int, default=N_clients, help="total users")
    parser.add_argument("--K_Layer_idx", nargs="*", type=int, default=K_Layer_idx, help="Model Layer Index")
    parser.add_argument("--alpha_factor", type=float, default=alpha_factor, help="alpha factor")
    args = parser.parse_args()

    return args
