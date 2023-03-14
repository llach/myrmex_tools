import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from myrmex_tools import merge_left_right, single_frame_heatmap, random_shift_seq

samples = []
datadir = f"{os.environ['HOME']}/cloth/edge/"

# load "edge" samples from the cloth dataset
for fname in os.listdir(datadir):
    if fname in ["pics", ".DS_Store"]: continue # skip directory containing visulizations and macOS bloat
    with open(f"{datadir}{fname}", "rb") as f: samples.append(pickle.load(f)["mm"])

# show random variations of the input sample
for _ in range(5):
    fig, ax = plt.subplots(ncols=1, figsize=0.5*np.array([10,9]))

    # translate sample randomly
    s = random_shift_seq([samples[1]], [True, True])

    # overlay both myrmex frames and plot heatmap
    merged = merge_left_right(s)
    single_frame_heatmap(merged, fig, ax)

    fig.tight_layout()
    plt.show()