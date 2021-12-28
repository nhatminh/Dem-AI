"FIX PROGRAM SETTINGS"
DEMLEARN_SETTING = True
READ_DATASET = True  # True or False => Set False to generate dataset.
PLOT_PATH = "./figs/"
RS_PATH = "./results/"

# N_clients = 50
N_clients = 50
DATASETS= ["mnist","fmnist","femnist","Cifar10"]
DATASET = DATASETS[0] ### don't need to input dataset from options

# RUNNING_ALGS = ["FedAvg","fedprox","demlearn","demlearn-p","demlearn-rep","demlearnp-rep","pFedMe"]  #use 3 for DemLearn
RUNNING_ALGS = ["FedAvg","fedprox","demlearn-p","pFedMe"]
# ,choices=["pFedMe", "pFedMe_p", "PerAvg", "FedAvg", "FedU", "Mocha", "Local" , "Global","DemLearn","DemLearnRep"])
RUNNING_ALG = RUNNING_ALGS[2] #use 2 for DemLearn

### Agorithm Parameters ###
Amplify = True ## trick to applify initial steps

if(DATASET == "mnist"):
    NUM_GLOBAL_ITERS = 100
    PARAMS_mu = 0.002 # 0.005, 0.002, 0.001, 0.0005  => choose 0.002  for demlearn
    alpha_factor = 0.6 #0.5 0.6 0.7 0.8 => choose 0.6
    Learning_rate = 0.05

elif(DATASET == "fmnist"):
    NUM_GLOBAL_ITERS = 100
    PARAMS_mu = 0.002 # 0.005, 0.002, 0.001, 0.0005  => select 0.002
    alpha_factor = 0.7  # 0.5 0.6 0.7 0.8 => choose 0.7
    Learning_rate = 0.05

elif(DATASET == "femnist"):
    NUM_GLOBAL_ITERS = 100
    PARAMS_mu = 0.002  # 0.005, 0.002, 0.001, 0.0005  => select 0.002
    alpha_factor = 0.7  # 0.5 0.6 0.7 0.8 => choose 0.7
    Learning_rate = 0.05

elif(DATASET == "Cifar10"):
    # N_clients = 20
    NUM_GLOBAL_ITERS = 100
    PARAMS_mu = 0.002  # 0.005, 0.002, 0.001, 0.0005 => select 0.002
    alpha_factor = 0.7
    Learning_rate = 0.05

if (RUNNING_ALG =="demlearn"):
    PARAMS_mu = 0

if(N_clients == 100):
    PARAMS_mu = 0.0005

LOCAL_EPOCH = 2

# PARAMS_mu = 1.0
K_Levels = 3  #1, 2, 3  # plus root level => K = K+1 in paper
# K_Layer_idx= [3,2,1]   #mean: we start at layer 1, 2, 3  for each level
K_Layer_idx= [4,3,1]   #mean: we start at layer 1, 3, 4  for each level (2 CNNs, 2 FCs, each layer has weights and bias)
# K_Layer_idx= [5,4,1]   #mean: we start at layer 1, 4, 5  for each level (2 CNNs, 3 FCs, each layer has weights and bias)
# K_Layer_idx= [4,2,1]   #mean: we start at layer 1, 2, 4  for each level
TREE_UPDATE_PERIOD = 1 # tested with 1, 2, 3, NUM_GLOBAL_ITERS => TREE_UPDATE_PERIOD = NUM_GLOBAL_ITERS: fixed hierrachical structure from beginning
CLUSTER_METHOD = "weight" #"weight" or "cosine"
MODEL_TYPE = "cnn" #"cnn" or "mclr"


PARAMS_gamma = 1.0   # Hard Update for DemLearn
# PARAMS_gamma = 0.9    # Soft Update for DemLearn-Rep => a bit slower but converge to better solution
# PARAMS_beta = 0.1     # Use in DemLearn-Rep algorithm (allow small value for personalization)
# PARAMS_beta = 1.    # Use in DemLearn algorithm and will be decay
PARAMS_beta = 1.    # Use in DemLearn algorithm and will be decay
DECAY = False # True or False =>  Decay of beta parameter

if "dem" in RUNNING_ALG:
    # rs_file_path = "{}_I{}_K{}_T{}_b{}_a{}_m{}_{}.h5".format(
    #     DATASET, NUM_GLOBAL_ITERS, K_Levels, TREE_UPDATE_PERIOD,
    #     str(PARAMS_beta).replace(".", "-"), alpha_factor, str(PARAMS_mu).replace(".", "-"), CLUSTER_METHOD[0])
    if(Amplify):
        rs_file_path= "{}_I{}_K{}_T{}_b{}_a{}_m{}_{}.h5".format(
            DATASET, NUM_GLOBAL_ITERS, K_Levels, TREE_UPDATE_PERIOD,
            str(PARAMS_beta).replace(".","-"), alpha_factor, str(PARAMS_mu).replace(".","-"), CLUSTER_METHOD[0])
    else:
        rs_file_path= "{}_I{}_K{}_T{}_b{}_a{}_m{}_{}_A{}.h5".format(
            DATASET, NUM_GLOBAL_ITERS, K_Levels, TREE_UPDATE_PERIOD,
            str(PARAMS_beta).replace(".","-"), alpha_factor, str(PARAMS_mu).replace(".","-"), CLUSTER_METHOD[0],Amplify)

    print("Result Path: ", rs_file_path)
else:
    rs_file_path = "{}_{}_I{}.h5".format(RUNNING_ALG,DATASET,NUM_GLOBAL_ITERS)
    print("Result Path: ", rs_file_path)

# complex_file_path = "{}_{}_I{}_time.h5".format(DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS)





