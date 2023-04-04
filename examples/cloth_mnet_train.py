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
labels  = np.concatenate([len(value)*[classes.index(key)] for key, value in dataset.items()])
labels  = torch.tensor(F.one_hot(torch.tensor(labels, dtype=torch.int64)), dtype=torch.float32)

####
## HYPERPARAMETERS
####
TRAIN_RATIO = 0.8
BATCH_SIZE  = 16
N_EPOCHS    = 40
PRINT_FREQ  = 10

# train-test-split
tensor_ds = TensorDataset(torch.tensor(inputs, dtype=torch.float32), labels)
N_train = int(len(tensor_ds)*TRAIN_RATIO)
N_test = len(tensor_ds)-N_train
train_ds, test_ds = random_split(
    tensor_ds, 
    [N_train, N_test]
)
trainloader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
testloader  = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE)

# create network, loss and optimizer
mnet = MyrmexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    mnet.parameters(), 
    lr=1e-3,
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=0, 
    amsgrad=False
)

# training loop
for epoch in range(N_EPOCHS):  # loop over the dataset multiple times

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
        if i % PRINT_FREQ == PRINT_FREQ-1: # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_FREQ:.5f}')
            running_loss = 0.0

print('Finished Training')