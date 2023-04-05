import os
import torch
import pickle

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader

from datetime import datetime
from myrmex_tools import full_augment, flatten_batch, MyrmexNet

def plot_learning_curve(train_loss, test_loss, ax, min_test=None, min_test_i=0):

    # generate xs 
    xs = np.arange(len(train_loss)).astype(int)+1

    # calculate mean and 95%-percentiles for the test loss 
    tl_mean = np.mean(test_loss, axis=1)
    tl_upper = np.percentile(test_loss, 0.95, axis=1)
    tl_lower = np.percentile(test_loss, 0.05, axis=1)

    # plot train and (mean) test loss, give train and test loss in legend of batch with lowest test loss
    ax.plot(xs, train_loss, label=f"training loss | {train_loss[min_test_i]:.5f}")
    ax.plot(xs, tl_mean, label=f"test loss | {tl_mean[min_test_i]:.5f}")

    # annotate test loss percentiles
    ax.fill_between(xs, tl_mean+tl_upper, tl_mean-tl_lower, color="#A9A9A9", alpha=0.3, label="test loss 95%ile")

    # annotate best test loss
    ax.scatter(min_test_i, min_test, c="red", marker="X", linewidths=0.7, label="best avg. test loss")

    ax.set_xlabel("#batches")
    ax.set_ylabel("loss [cross-entropy]")
    ax.set_title("Myrmex Grasp Type Classification Network")
    ax.legend()

def test_net(model, crit, dataset):
    """ we evaluate the model on the whole test set and return outputs, labels and losses
    """
    losses = []
    outputs = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data in dataset:
            inputs, lbls = data
            outs = model(inputs)
            loss_t = crit(outs, lbls)

            losses.append(loss_t.numpy())
            outputs.append(outs.numpy())
            labels.append(lbls.numpy())
    model.train()
    return np.concatenate(outputs, axis=0), np.concatenate(labels, axis=0), np.concatenate(losses, axis=0)

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
STORE_PATH  = f"{os.environ['HOME']}/cloth_trainings/"
TRAIN_RATIO = 0.8
BATCH_SIZE  = 16
N_EPOCHS    = 40
N_TEST_AVG  = 5

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

# create folder structure to store weights and metadata
trial_path   = f"{STORE_PATH}/mnet_{datetime.now().strftime('%Y.%m.%d_%H-%M-%S')}"
weights_path = f"{trial_path}/weights"
for p in [STORE_PATH, trial_path, weights_path]: os.makedirs(p, exist_ok=True)

train_losses     = []
train_accuracies = []

min_test_loss = np.inf 
test_losses   = []

# code adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
nbatch = 0
for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mnet(inputs)
        loss = torch.mean(criterion(outputs, labels))
        loss.backward()
        optimizer.step()

        # store performance stats
        train_loss = loss.item()
        train_losses.append(train_loss)

        # compute test loss on whole test set
        test_out, test_lbl, test_loss = test_net(mnet, nn.CrossEntropyLoss(reduction="none"), testloader)
        test_losses.append(test_loss)

        # new best average test loss? then store the weights
        test_avg = np.mean(test_losses[-N_TEST_AVG:])
        if len(test_losses) >= N_TEST_AVG and test_avg < min_test_loss:
            print(f"new best model with {test_avg:.5f}")
            torch.save(mnet.state_dict(), f"{weights_path}/best.pth")
            min_test_loss   = test_avg
            min_test_loss_i = nbatch

        nbatch += 1
        print(f"[{epoch + 1}, {i + 1:5d}] train: {train_loss:.3f} | test: {np.mean(test_loss):.3f}", end="")
        print()

    # store model weights after every epoch
    torch.save(mnet.state_dict(), f"{weights_path}/epoch_{epoch+1}.pth")
# store final weights
torch.save(mnet.state_dict(), f"{weights_path}/final.pth")

# store losses in case we want prettier plots later
with open(f"{trial_path}/losses.pkl", "wb") as f:
    pickle.dump({
        "train": train_losses,
        "test": test_losses,
        "min_test": [min_test_loss, min_test_loss_i]
    }, f)

fig, ax = plt.subplots()
plot_learning_curve(train_losses, test_losses, ax, min_test_loss, min_test_loss_i)

fig.tight_layout()
fig.savefig(f"{trial_path}/learning_curve.png")

plt.clf()
plt.close()

print('Finished Training')