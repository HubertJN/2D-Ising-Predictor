import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
import warnings
import sys
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams["figure.dpi"] = 120

# colors
soft_red = "#ed9688"
soft_blue = "#78b3eb"
soft_green = "#9befdf"

data = np.load("./plotting_data/prediction_actual_gnn.npy")
train_loss = np.load("./plotting_data/train_loss_gnn.npy")
val_loss = np.load("./plotting_data/val_loss_gnn.npy")

line = np.linspace(np.min(data[:,0]),np.max(data[:,0]),10)

# sort data for easier plotting
data = data[data[:, 0].argsort()]

alpha = data[:,1]
beta = data[:,2]
expectation = alpha/(alpha+beta)
rmse = np.sqrt(np.mean((data[:,0]-expectation)**2))
variance = np.sqrt(alpha*beta/((alpha+beta)**2*(alpha+beta+1)))

spacing = np.linspace(np.min(data[:,0]),np.max(data[:,0]), 101)
std_dev = np.zeros(100)
mean = np.zeros(100)

for i in range(100):
    index = np.where(np.logical_and(mean >= spacing[i], mean < spacing[i+1]))

    std_dev[i] = np.std(expectation[index])
    mean[i] = np.mean(expectation[index])

#spacing = np.delete(spacing, -1)
plt.plot(line, line, color=soft_red)
plt.scatter(data[:,0], expectation, s = 1, color=soft_blue)
#plt.fill_between(spacing, mean+std_dev, mean-std_dev, alpha=0.3, edgecolor=soft_blue, facecolor=soft_blue)
plt.title("Neural Network Prediction Assessment")
plt.xlabel("Target")
plt.ylabel("Prediction")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "RMSE: {:.5f}".format(rmse), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.savefig("figures/gnn_tweaking/prediction_target.pdf", bbox_inches="tight")
plt.show()
plt.close()

plt.plot(train_loss, label="Train Loss", color=soft_blue)
plt.plot(val_loss, label="Validation Loss", color=soft_red)
plt.title("Neural Network Training")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("figures/gnn_tweaking/loss.pdf", bbox_inches="tight")
plt.show()
plt.close()
