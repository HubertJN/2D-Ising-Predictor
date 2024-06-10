import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
import warnings
import sys
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

show_plot = False
if len(sys.argv) > 2:
    show_plot = True

if len(sys.argv) > 1:
    run = int(sys.argv[1])
else:
    print("Error. No run parameter. Exiting.")
    exit()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams["figure.dpi"] = 120

# colors
soft_red = "#ed9688"
soft_blue = "#78b3eb"
soft_green = "#9befdf"

data = np.load("./plotting_data/gnn/gnn_prediction_actual_%d.npy" % run)
train_loss = np.load("./plotting_data/gnn/gnn_train_loss_%d.npy" % run)
val_loss = np.load("./plotting_data/gnn/gnn_val_loss_%d.npy" % run)

line = np.linspace(np.min(data[:,0]),np.max(data[:,0]),10)

# average loss for better plot
train_loss = np.mean(train_loss.reshape(-1, 10), axis=1)
val_loss = np.mean(val_loss.reshape(-1, 10), axis=1)

# sort data for easier plotting
data = data[data[:, 0].argsort()]

alpha = data[:,1]
beta = data[:,2]
expectation = alpha/(alpha+beta)
rmse = np.sqrt(np.mean((data[:,0]-expectation)**2))
variance = np.sqrt(alpha*beta/((alpha+beta)**2*(alpha+beta+1)))

num_top = 5
error = abs(data[:,0]-expectation)
top_max = np.argpartition(error, -num_top)[-num_top:]

plt.plot(line, line, color=soft_red)
plt.scatter(data[:,0], expectation, s=1, color=soft_blue)
plt.scatter(data[:,0][top_max], expectation[top_max], s=60, facecolors='none', edgecolors=soft_red)
plt.title("Neural Network Prediction Assessment")
plt.xlabel("Target")
plt.ylabel("Prediction")
ax = plt.gca()

for i, index in enumerate(top_max):
    ax.annotate("%d" % (i+1), (data[:,0][index], expectation[index]), (data[:,0][index]+0.01, expectation[index]+0.01), fontsize=10)

ax.set_box_aspect(1)
plt.text(0.05, 0.95, "RMSE: {:.5f}".format(rmse), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.savefig("figures/gnn/prediction_target_%d.pdf" % run, bbox_inches="tight")
if show_plot == True:
    plt.show()
plt.close()

plt.plot(10*np.arange(train_loss.size), train_loss, label="Train Loss", color=soft_blue)
plt.plot(10*np.arange(val_loss.size), val_loss, label="Validation Loss", color=soft_red)
plt.title("Neural Network Training")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("figures/gnn/loss_%d.pdf" % run, bbox_inches="tight")
if show_plot == True:
    plt.show()
plt.close()

with open("plotting_data/gnn/hyperparameters_%d.txt" % run, 'a') as f:
    f.write('%s = %s\n' % ('rmse', rmse))
