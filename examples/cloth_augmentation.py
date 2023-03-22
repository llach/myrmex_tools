import os
import pickle

from myrmex_tools import merge_left_right

dataset = {}
datadir = f"{os.environ['HOME']}/cloth/"

# iterate over class folders
for label in os.listdir(datadir):
    if label == ".DS_Store": continue
    dataset.update({label: []})

    for fname in os.listdir(f"{datadir}{label}"):
        if fname in ["pics", ".DS_Store"]: continue # skip directory containing visulizations and macOS bloat
        with open(f"{datadir}{label}/{fname}", "rb") as f: dataset[label].append(pickle.load(f)["mm"])

# merge all samples of each class
for label, samples in dataset.items():
    dataset[label] = merge_left_right(samples)
