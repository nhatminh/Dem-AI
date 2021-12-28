# DemLearn Algorithm Implementation

The detail of algorithm was introduced in our recent paper:
*"Self-organizing Democratized Learning: Towards Large-scale Distributed Learning Systems",* [Paper Link](https://arxiv.org/abs/2007.03278)

We are working for the demonstration of some properties and initially develop new algorithm for Democratized Learning systems based on our Dem-AI philosophy paper
*"Distributed and Democratized Learning: Philosophy and Research Challenges",* [Paper Link](https://arxiv.org/abs/2003.09301)


This code is forked and reimplemented based on our prior paper and pFedMe code:

*"Federated Learning over Wireless Networks: Convergence Analysis and Resource Allocation"*
[Paper Link](https://arxiv.org/abs/1910.13067)

pFedMe implementation as: https://github.com/CharlieDinh/pFedMe

Running Instruction
======
 In the current version, we evaluate based on **MNIST** and **Fashion-MNIST** Dataset.
 
 **Environment:** Python 3.8, Pytorch
 
 **Downloading dependencies**

```
pip3 install -r requirements.txt  
```

 **Main File:** `demlearn_main.py`,  **Setting File:** `Seting.py`,  **Plot Summary Results:** `dem_plot_summary.py`, 
 
 **PATH:** Output: `./results`, Figures: `./figs`, Dataset: `./data`

 **Generated Dataset:** [Google Drive Link](https://drive.google.com/drive/folders/1qhVuh5S_UIr9U5SdULzr0l79_EsdsX8E?usp=sharing),
 Unzip each dataset file and move to ./data/mnist/data or ./data/fmnist/data 
 
### Input Parameters for the algorithm:
The more detail setting can be found in `Setting.py` file

- Four algorithms: *"fedavg","fedprox","demlearn"*
- Datasets: *"mnist", "fmnist", "femnist", "Cifar10"*
- Clustering metrics: *"weight"*, *"cosine"*
- Model: *"cnn"*
- Result file name: *"{}_ {}_ I{}_ K{}_ T{}_ b{}_ d{}_ m{}_ {}.h5"* that reflects DATASET, RUNNING_ALG, NUM_GLOBAL_ITERS, K_Levels, TREE_UPDATE_PERIOD, beta, DECAY, mu, cluster_metric(w or g).
    * E.g.: `fmnist_demlearn_I100_K1_T2_b1-0_dTrue_m0-001_w.h5`, `mnist_demlearn-p_I60_K1_T60_b1-0_dTrue_m0-002_w.h5`

To run the code with the `Setting.py` we use following command:
- `python demlearn_main.py`

### Comparison DemLearn vs FedAvg, FedProx
In particular, we conduct evaluations for specialization (C-SPE) and generalization (C-GEN) of learning models at agents on average that are defined as the performance in their local test data only, and the collective test data from all agents in the region, respectively. Accordingly, we denote Global as global model performance; G-GEN, G-SPE are the average generalization and specialization performance of group models, respectively.

#### MNIST DATASET with 50 users
![](https://github.com/nhatminh/Dem-AI/blob/master/figs/mnist_dem_vs_fed.png)

#### FASHION-MNIST DATASET with 50 users
![](https://github.com/nhatminh/Dem-AI/blob/master/figs/fmnist_dem_vs_fed.png)


#### FEDERATED EXTENDED MNIST DATASET with 50 users
![](https://github.com/nhatminh/Dem-AI/blob/master/figs/femnist_dem_vs_fed.png)

#### CIFAR-10 DATASET with 50 users
![](https://github.com/nhatminh/Dem-AI/blob/master/figs/femnist_dem_vs_fed.png)