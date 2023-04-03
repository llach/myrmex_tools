import os
import torch
import pickle
import numpy as np
import torch.optim as optim

from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader

from myrmex_tools import full_augment, flatten_batch, MyrmexNet

dataset = {}
datadir = f"{os.environ['HOME']}/cloth/"

"""
STEP I: read data
"""
# iterate over class folders
for label in os.listdir(datadir):
    if label == ".DS_Store": continue
    dataset.update({label: []})

    for fname in os.listdir(f"{datadir}{label}"):
        if fname in ["pics", ".DS_Store"]: continue # skip directory containing visulizations and macOS bloat
        with open(f"{datadir}{label}/{fname}", "rb") as f: dataset[label].append(pickle.load(f)["mm"])

"""
STEP II: preprocess data
"""
# OPTION I) merge all samples of each class
# for label, samples in dataset.items():
#     dataset[label] = merge_left_right(samples)

# OPTION II) treat left/right as individual samples → double the number of samples
for label, samples in dataset.items():
    samples = np.array(samples)
    samples[:,1,:] = np.flip(samples[:,1,:], axis=2) # we still align left and right
    dataset[label] = flatten_batch(samples)

"""
STEP III: augment data
"""

# augment dataset
print("dataset size:")
for label, samples in dataset.items():
    dataset[label] = flatten_batch([full_augment(s) for s in samples])
    print(f"\t{label}: {len(dataset[label])}")

"""
STEP IV: prepare Torch dataset, create network
"""
# classes are the dict's keys
classes = list(dataset.keys())

# stack inputs, add channel dimension for conv2d
inputs  = np.vstack([v for v in dataset.values()])
inputs  = np.expand_dims(inputs, axis=1)

# generate class label vector, convert to indices, then to one-hot
labels  =  np.concatenate([len(value)*[classes.index(key)] for key, value in dataset.items()])
labels = torch.tensor(F.one_hot(torch.tensor(labels, dtype=torch.int64)), dtype=torch.float32)

# train-test-split
tensor_ds = TensorDataset(torch.tensor(inputs, dtype=torch.float32), labels)

N_train = int(len(tensor_ds)*0.8)
N_test = len(tensor_ds)-N_train
train_ds, test_ds = random_split(
    tensor_ds, 
    [N_train, N_test]
)
trainloader = DataLoader(train_ds, shuffle=True, batch_size=16)
testloader  = DataLoader(test_ds, shuffle=False, batch_size=16)

# create network, loss and optimizer
mnet = MyrmexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mnet.parameters(), lr=0.001, momentum=0.9)

# training loop
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9: # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
            running_loss = 0.0

print('Finished Training')