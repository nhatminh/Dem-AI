#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
from utils.model_utils import read_data

torch.manual_seed(0)


def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu):
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    for i in range(times):
        print("---------------Running time:------------", i)
        # Generate model
        if (model == "mclr"):
            if (dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            else:
                model = Mclr_Logistic(60, 10).to(device), model

        if (model == "cnn"):
            if (dataset == "Mnist"):
                model = Net().to(device), model
            elif (dataset == "Cifar10"):
                model = CifarNet().to(device), model

        if (model == "dnn"):
            if (dataset == "Fmnist_DemAI" or dataset == "Mnist"):
                model = Mnist_DemAI().to(device), model
            elif (dataset == "femnist" or dataset == "femnist_DemAI"):
                model = femnist_DemAI().to(device), model
                print("Model: femnist_DemAI is applied")
            elif (dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model
                print("Model: Cifar10 CNN is applied")
            else:
                model = DNN(60, 20, 10).to(device), model

        # select algorithm
        if (algorithm == "FedAvg"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                            local_epochs, optimizer, numusers, i)

        if (algorithm == "pFedMe"):
            # server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
            #                 local_epochs, optimizer, numusers, K, personal_learning_rate, i, cutoff=0,times=1)
            data = read_data(dataset), dataset
            server = pFedMe(None, device, data, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, cutoff=0)

        if (algorithm == "PerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                            local_epochs, optimizer, numusers, i)

        server.train()
        server.test()

    # Average data
    if (algorithm == "PerAvg"):
        algorithm == "PerAvg_p"
    if (algorithm == "pFedMe"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                     learning_rate=learning_rate, beta=beta, algorithms="pFedMe_p", batch_size=batch_size,
                     dataset=dataset, k=K, personal_learning_rate=personal_learning_rate, times=times)
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                 learning_rate=learning_rate, beta=beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset,
                 k=K, personal_learning_rate=personal_learning_rate, times=times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10",
                        choices=["Mnist", "Fmnist_DemAI", "Mnist_DemAI", "Synthetic", "Cifar10", "femnist",
                                 "femnist_DemAI"])
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.08,
                        help="Local learning rate")  # eta => use for update w and affect global model
    parser.add_argument("--beta", type=float, default=2.,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")  # default: 1
    parser.add_argument("--lamda", type=int, default=10
                        , help="Regularization term")  # default: 15
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=2)  # R rounds default is 10
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe", choices=["pFedMe", "Per'Avg", "FedAvg"])
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=2, ## ????? set parameters here
                        help="Computation steps")  # default: 5, k steps to find theta approximation (personalized model)
    parser.add_argument("--personal_learning_rate", type=float, default=0.04,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")  # inner problem (7) find personal model
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    # python3 main.py --dataset Mnist --model dnn --batch_size 10 --learning_rate 0.17 --num_global_iters 2 --local_epochs 2 --algorithm pFedMe --numusers 50 --times 1
    # python3 main.py --dataset Cifar10 --model dnn --algorithm pFedMe --times 1
    # For Mnist_DemAI, learning_rate = 0.1, personal_learning_rate=0.04,  beta from 2.0 , lamda 10, local epoch 2
    # For Fmnist_DemAI, learning_rate = 0.08, personal_learning_rate=0.04,  beta from 2.0 , lamda 10
    # For Femnist_DemAI, learning_rate = 0.25, personal_learning_rate=0.1,  beta=2.0 , lamda = 5....
    # For CIFAR-10_DemAI, learning_rate = 0.08, personal_learning_rate=0.04,  beta from 2.0 , lamda 10

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        gpu=args.gpu
    )
