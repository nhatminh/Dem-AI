from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)
NUM_USERS = 20
NUM_LABELS = 2
# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

# Get MNIST data, normalize, and divide by level
# mnist = fetch_openml('MNIST original', data_home='./data')
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings

mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
mnist_data = []
for i in trange(10):
    idx = mnist.target==i
    mnist_data.append(mnist.data[idx])

print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])

###### CREATE USER DATA SPLIT #######
# Assign 150 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):  # 3 labels for each users
        #l = (2*user+j)%10
        l = (user + j) % 10    # round robin style to assign labels keep user_index in 10 base or 100 base
        print("L:", l)
        X[user] += mnist_data[l][idx[l]:idx[l]+10].tolist()
        y[user] += (l*np.ones(10)).tolist()
        idx[l] += 5

print("IDX1:", idx)  # counting samples for each labels

# Assign remaining sample by power law, lognormal
user = 0
props = np.random.lognormal(0, 1.0, (10, NUM_USERS, NUM_LABELS))  #sigma = 1.0 or 2.0, how high of distribution.
props = np.array([[[len(v)-1000]] for v in mnist_data]) * props/np.sum(props, (1, 2), keepdims=True)
# print("here:",props/np.sum(props,(1,2), keepdims=True))
#props = np.array([[[len(v)-100]] for v in mnist_data]) * \
#    props/np.sum(props, (1, 2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
# print("here2:",props)
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # the same 3 labels for each users but new remaining data
        # l = (2*user+j)%10
        l = (user + j) % 10     # round robin style to assign labels keep user_index in 10 base or 100 base
        num_samples = int(props[l, user//int(NUM_USERS/10), j])
        num_samples = int(num_samples *0.2) #Scale down number of samples

        # numran1 = random.randint(1, 10)     # num_samples plus 50, 100, 200
        # numran2 = random.randint(1, 2)        # scale up num_samples to 2, 5, 10 times
        # # num_samples = (num_samples) * numran2 + numran1  #Scale up number of samples by factors of numran2, numran1
        # num_samples = num_samples  + numran1  # Scale up number of samples by factors of numran2, numran1
        if(NUM_USERS <= 10):
            num_samples = num_samples * 2
        # num_samples = int(props[l,user//int(NUM_USERS/10),j])
        # num_samples = min(num_samples,200)

        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print("check len os user:", user, j,
                  "len data", len(X[user]), num_samples)

print("IDX2:", idx) # counting samples for each labels
print("Remaining samples added:", sum(idx))
# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.8*num_samples)  #Test 80%
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Numb_training_samples:", train_data['num_samples'])
print("Total_training_samples:",sum(train_data['num_samples']))
print("Numb_testing_samples:", test_data['num_samples'])
print("Total_testing_samples:",sum(test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
