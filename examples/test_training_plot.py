import os 
import pickle
import matplotlib.pyplot as plt

from cloth_mnet_train import plot_learning_curve

with open(f"{os.environ['HOME']}/cloth_trainings/mnet_2023.04.11_10-32-31/losses.pkl", "rb") as f:
    losses = pickle.loads(f.read())

train_losses        = losses["train"]
test_losses         = losses["test"]
train_accuracies    = losses["train_acc"]
test_accuracies     = losses["test_acc"]
min_test_loss, min_test_loss_i = losses["min_test"]

fig, ax = plt.subplots(figsize=(10,8))
plot_learning_curve(train_losses, test_losses, train_accuracies, test_accuracies, ax, min_test_loss, min_test_loss_i)

fig.tight_layout()
plt.show()