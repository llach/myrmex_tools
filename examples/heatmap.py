import os
import torch
import numpy as np

from myrmex_tools import MyrmexNet
from cloth_mnet_train import load_and_preprocess_data, augment_dataset, create_dataloader

NCLASSES = 5
DATA_DIR   = f"{os.environ['HOME']}/cloth/"
MODEL_PATH = f"{os.environ['HOME']}/cloth_trainings/mnet_2023.04.11_10-32-31/"
# MODEL_PATH = f"{os.environ['HOME']}/cloth_trainings/mnet_2023.04.13_13-26-52/"

# instantiate model, load weights
model = MyrmexNet()
model.load_state_dict(torch.load(f"{MODEL_PATH}/weights/best.pth"))
model.eval()

samples, labels, classes = load_and_preprocess_data(DATA_DIR)

# augment dataset
# X, y = augment_dataset(samples, labels)
X, y = samples, labels
X = np.expand_dims(X, axis=1)

dataloader = create_dataloader(X, y, shuffle=False, batch_size=32)

confusion_matrix = np.zeros((NCLASSES, NCLASSES), dtype=np.int64)
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print(preds.view(-1), labels.view(-1))

        for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

import matplotlib.pyplot as plt
import seaborn as sn

fig, ax = plt.subplots(figsize=(9,8))
sn.heatmap(confusion_matrix, annot=True, fmt=",")

ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax.set_xticks(np.arange(len(classes))+0.5, classes)
ax.set_yticks(np.arange(len(classes))+0.5, classes)

fig.tight_layout()
plt.show()
pass