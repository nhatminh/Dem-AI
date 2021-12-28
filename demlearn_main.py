#!/usr/bin/env python
from comet_ml import Experiment
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
from FLAlgorithms.servers.serverFedU import FedU
from FLAlgorithms.servers.serverlocal import FedLocal
from FLAlgorithms.servers.serverglobal import FedGlobal
from FLAlgorithms.servers.serverDemLearn import DemLearn
from utils.model_utils import read_data
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
from utils.options import args_parser


# import comet_ml at the top of your file
# python3 demlearn_main.py  --algorithm DemLearn   #use this


#python3 demlearn_main.py --dataset Mnist --model cnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 3 --algorithm DemLearn  --time 1 --subusers 1.0
#python3 demlearn_main.py --dataset Mnist --model mclr --learning_rate 0.03 --num_global_iters 200  --algorithm FedAvg --times 10 --subusers 1.0
# Create an experiment with your api key:
def main(experiment, dataset, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, commet, gpu, cutoff, args):

    # print torch.device()
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    data = read_data(dataset) , dataset

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        # print(">>>> here2:", model)
        if(model == "mclr"):
            if(dataset == "human_activity"):
                model = Mclr_Logistic(561,6).to(device), model
            elif(dataset == "gleam"):
                model = Mclr_Logistic(561,6).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = Mclr_Logistic(100,2).to(device), model
            elif(dataset == "Synthetic"):
                model = Mclr_Logistic(60,10).to(device), model
            elif(dataset == "EMNIST"):
                model = Mclr_Logistic(784,62).to(device), model
            else:#(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model

        elif(model == "dnn"):
            if(dataset == "human_activity"):
                model = DNN(561,100,12).to(device), model
            elif(dataset == "gleam"):
                model = DNN(561,20,6).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = DNN(100,20,2).to(device), model
            elif(dataset == "Synthetic"):
                model = DNN(60,20,10).to(device), model
            elif(dataset == "EMNIST"):
                model = DNN(784,200,62).to(device), model
            else:#(dataset == "Mnist"):
                model = DNN2().to(device), model
        
        elif(model == "cnn"):
            # if(dataset == "Fmnist_DemAI" or dataset == "Mnist_DemAI"):
            #     model = Net_DemAI().to(device), model
            if (dataset == "femnist"):
                model = femnist_DemAI().to(device), model
            elif(dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model
            else:#(dataset == "Mnist"):
                model = Net_DemAI().to(device), model

        # select algorithm
        # print(">>>> here1")
        if(algorithm == "FedAvg" or algorithm == "fedprox" ):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedAvg(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff, args)
        
        elif(algorithm == "PerAvg"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(personal_learning_rate) + "_" + str(learning_rate)+  "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = PerAvg(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff)

        elif(algorithm == "FedU"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate)+ "_" + str(L_k) + "L_K"+ "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedU(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, K, i, cutoff)

        elif(algorithm == "pFedMe"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(personal_learning_rate) +  "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = pFedMe(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, cutoff)
        elif (algorithm == "DemLearn"):
            if (commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(personal_learning_rate) + "_" + str(num_glob_iters) + "_" + str(local_epochs) + "_" + str(numusers))
            server = DemLearn(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, cutoff, args)
        elif(algorithm == "Local"):
            if(commet): 
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(L_k) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedLocal(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff)

        elif(algorithm == "Global"):
            if(commet): 
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(L_k) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedGlobal(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i, cutoff)
        else:
            print("Algorithm is invalid")
            return
        # print(">>>> here")
        server.train()
        # server.test()

    # average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L_k,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times, cutoff = cutoff)

if __name__ == "__main__":
    args = args_parser()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.subusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="Xmop6IBZEzIWbCjme4I9vi0tB",
            project_name="demlearn",
            workspace="nhatminh",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "model":args.model,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "beta" : args.beta, 
            "L_k" : args.L_k,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "optimizer": args.optimizer,
            "numusers": args.subusers,
            "K" : args.K,
            "personal_learning_rate" : args.personal_learning_rate,
            "times" : args.times,
            "gpu": args.gpu,
            "cut-off": args.cutoff
        }
        
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        L_k = args.L_k,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.subusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        commet = args.commet,
        gpu=args.gpu,
        cutoff = args.cutoff,
        args = args
        )

