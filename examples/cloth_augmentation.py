import os
import pickle
import numpy as np

from myrmex_tools import merge_left_right, single_frame_heatmap, full_augment, flatten_batch

dataset = {}
datadir = f"{os.environ['HOME']}/cloth/"

# iterate over class folders
for label in os.listdir(datadir):
    if label == ".DS_Store": continue
    dataset.update({label: []})

    for fname in os.listdir(f"{datadir}{label}"):
        if fname in ["pics", ".DS_Store"]: continue # skip directory containing visulizations and macOS bloat
        with open(f"{datadir}{label}/{fname}", "rb") as f: dataset[label].append(pickle.load(f)["mm"])

# OPTION I) merge all samples of each class
# for label, samples in dataset.items():
#     dataset[label] = merge_left_right(samples)

# OPTION II) treat left/right as individual samples → double the number of samples
for label, samples in dataset.items():
    samples = np.array(samples)
    samples[:,1,:] = np.flip(samples[:,1,:], axis=2) # we still align left and right
    dataset[label] = flatten_batch(samples)

# plot data to check whether dataset construction was done correctly
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,5))

# for i, sample in enumerate(dataset["edge"]):
#     ridx, cidx = i%2, int(i/2)
#     single_frame_heatmap(sample, fig, axes[ridx,cidx], with_colorbar=False)
#     if ridx==0: axes[ridx,cidx].set_title(f"sample {cidx+1}")

# axes[0,0].set_ylabel("left")
# axes[1,0].set_ylabel("right")

# fig.tight_layout()
# plt.show()

# augment dataset
for label, samples in dataset.items():
    dataset[label] = flatten_batch([full_augment(s) for s in samples])