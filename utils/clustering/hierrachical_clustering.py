import numpy as np

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from Setting import *
from utils.clustering.DTree import Node

''' sklearn.cluster.AgglomerativeClustering
linkage{“ward”, “complete”, “average”, “single”}, default=”ward”
Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. 
The algorithm will merge the pairs of cluster that minimize this criterion.
. ward minimizes the variance of the clusters being merged.
. average uses the average of the distances of each observation of the two sets.
. complete or maximum linkage uses the maximum distances between all observations of the two sets.
. single uses the minimum of the distances between all observations of the two sets.


affinity: str or callable, default=’euclidean’
Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. 
If linkage is “ward”, only “euclidean” is accepted. 
If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
'''
## results of AgglomerativeClustering
'''children_:The children of each non-leaf node.  Values less than n_samples correspond to leaves of the tree which are the original samples. 
A node i greater than or equal to n_samples is a non-leaf node and has children is children_[i - n_samples]. 
Alternatively at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i'''



def cal_linkage_matrix(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)  # 150 samples
    for i, merge in enumerate(model.children_):
        # print(i,merge)
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:  # leaf node
                current_count += 1
            else:  # cluster head or merged node
                current_count += counts[child_idx - n_samples]  # count all nodes belongs to this head ??? strange form?
        counts[i] = current_count
        # print(counts)

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    return n_samples, linkage_matrix


def retrieve_leaves(model, cluster_head):
    leaves = []
    if (cluster_head < N_clients):
        leaves.append(cluster_head)
    else:
        for idx in model.children_[cluster_head - N_clients]:
            leaves = leaves + retrieve_leaves(model, idx)
    return leaves

def retrieve_cluster_head(model, cluster_head):
    if (cluster_head >= N_clients):
        return model.children_[cluster_head - N_clients]

def create_nodes(model, Parent, Clients):
    if (Parent.level == 1):
        childs_idx = retrieve_leaves(model, Parent["_id"])
        childs = []
        for c in childs_idx:
            child = Clients[c]
            child.parent = Parent
            childs.append(child)
        Parent.childs = childs
    else:
        cluster_heads = model.children_[Parent["_id"] - N_clients]
        Child1 = Node(_id=cluster_heads[0], parent=Parent, level=Parent.level - 1)
        Child2 = Node(_id=cluster_heads[1], parent=Parent, level=Parent.level - 1)
        if (cluster_heads[0] < N_clients):
            Child1 = Clients[cluster_heads[0]]
            Child1.parent = Parent
        else:
            create_nodes(model,Child1,Clients)
        if (cluster_heads[1] < N_clients):
            Child2 = Clients[cluster_heads[1]]
            Child2.parent = Parent
        else:
            create_nodes(model,Child2,Clients)

        Childs = [Child1, Child2]
        Parent.childs = Childs

def iris_clustering():
    iris = load_iris()
    X = iris.data
    print("Iris Data Shape:",iris.data.shape)

    # setting distance_threshold=0 ensures we compute the full tree.
    result = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    ### OR Plot Only get labels
    # model1 = result.fit_predict(X)
    # print(model1)

    # OR Plot
    model = result.fit(X)
    print("After Clustering:",model.labels_)
    return model

def weight_clustering(X):
    # setting distance_threshold=0 ensures we compute the full tree.
    #linkage {“ward”, “complete”, “average”, “single”}, default =”ward”
    #affinity {default: "euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”}
    # result = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity=None, linkage='average')
    result = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="euclidean", linkage='average')
    # result = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="cosine", linkage='average')

    ### OR Plot Only get labels
    # model1 = result.fit_predict(X)
    # print(model1)

    # OR Plot
    model = result.fit(X)
    # print("After Clustering:",model.labels_)
    return model

def cosine_clustering(X):
    # setting distance_threshold=0 ensures we compute the full tree.
    #linkage {“ward”, “complete”, “average”, “single”}, default =”ward”
    #affinity {default: "euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”}
    result = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="cosine", linkage='average')

    ### OR Plot Only get average
    # model1 = result.fit_predict(X)
    # print(model1)

    # OR Plot
    model = result.fit(X)
    # print("After Clustering:",model.labels_)
    return model




# def iris_test(model):
#     # print(model.children_)
#     # print(model.distances_)
#     numb_samples, rs_linkage_matrix = cal_linkage_matrix(model)
#     k_Levels = 2
#     for level in range(1, 2):
#         print("=> GENERALIZED LEVEL", k_Levels - level, "-------")
#         rs_dendrogram = dendrogram(rs_linkage_matrix, truncate_mode='level', p=level)
#         cluster_heads = rs_dendrogram['leaves']
#
#         for g_idx in cluster_heads:
#             if (g_idx < numb_samples):
#                 print("Leaf:", g_idx)
#                 # Node(g_idx,)
#             else:
#                 if (level == k_Levels - 1):
#                     all_leaves = retrieve_leaves(model, g_idx, numb_samples)
#                     print("All leaves:", all_leaves, " => Len:",len(all_leaves))
#                     # c_id = retrieve_leaves(g_idx, numb_samples)
#                 else:
#                     # h_id = retrieve_cluster_head(g_idx, numb_samples)
#                     print("Children:", retrieve_cluster_head(model, g_idx, numb_samples))


def tree_construction(model, Clients,round=0):
    # plot_dendrogram(cal_linkage_matrix(model)[1], round, alg)
    # print(model.labels_)
    # print(model.children_)
    cluster_heads = model.children_[-1]
    # print(cluster_heads)
    Root = Node(_id="Root", parent=None, level=K_Levels+1)
    Child1 = Node(_id=cluster_heads[0], parent=Root, level=K_Levels)
    Child2 = Node(_id=cluster_heads[1], parent=Root, level=K_Levels)
    if (cluster_heads[0] < N_clients):
        Child1 = Clients[cluster_heads[0]]
        Child1.parent = Root
    else:
        create_nodes(model, Child1, Clients)
    if (cluster_heads[1] < N_clients):
        Child2 = Clients[cluster_heads[1]]
        Child2.parent = Root
    else:
        create_nodes(model, Child2, Clients)

    Childs = [Child1, Child2]
    Root.childs = Childs

    return Root


if __name__ == "__main__":
    K_Levels = 3
    N_clients = 50
    Weight_dimension = 10

    Clients = []
    for c in range(N_clients):
        Clients.append(Node(_id=c, _type="Client", level=0))


    weights_matrix = np.random.rand(N_clients, Weight_dimension)
    model = weight_clustering(weights_matrix)
    gradient_matrix = np.random.rand(N_clients, Weight_dimension)
    model = gradient_clustering(gradient_matrix)

    Tree_Root = tree_construction(model, Clients)
    # print(Tree_Root.childs[0].childs[0].childs[0].childs)
    # print(Tree_Root.childs[0].childs)
    # for n in Clients:
    #     print(n)

    print("Number of agents in tree:", Tree_Root.count_clients())
    print("Number of agents in level K:",Tree_Root.childs[0].count_clients(), Tree_Root.childs[1].count_clients())
    # print("Number of agents Group 1 in level K-1:",Tree_Root.childs[0].childs[0].count_clients(), Tree_Root.childs[0].childs[1].count_clients())

    # model = iris_clustering()
    # N_clients = 150
    # iris_test(model)