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
if len(sys.argv) > 3:
    show_plot = True

if len(sys.argv) > 1:
    net_type = sys.argv[1].lower()
    run = int(sys.argv[2])
else:
    print("Error. No run parameter. Exiting.")
    exit()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 25
plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams["figure.dpi"] = 120

# colors
soft_red = "#ed9688"
soft_blue = "#78b3eb"
soft_green = "#9befdf"

data = np.load("./plotting_data/%s/%s_prediction_actual_%d.npy" % (net_type, net_type, run))
train_loss = np.load("./plotting_data/%s/%s_train_loss_%d.npy" % (net_type, net_type, run))
val_loss = np.load("./plotting_data/%s/%s_val_loss_%d.npy" % (net_type, net_type, run))

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

num_select = 3
error = data[:,0]-expectation
over_index = np.argpartition(error, num_select)[:num_select]
under_index = np.argpartition(error,- num_select)[-num_select:]
ideal_index = np.argpartition(abs(error), num_select)[:num_select]
combined_index = np.concatenate([over_index, ideal_index, under_index])

index_dict = {
    "over" : over_index,
    "ideal" : ideal_index,
    "under" : under_index
}

plt.plot(line, line, color=soft_red)
plt.scatter(data[:,0], expectation, s=1, color=soft_blue)
plt.scatter(data[:,0][combined_index], expectation[combined_index], s=60, facecolors='none', edgecolors=soft_red)
plt.xlabel("Target")
plt.ylabel("Prediction")
ax = plt.gca()

i = 0
index_type = ["over", "ideal", "under"]
for index_string in index_type:
    index_array = index_dict[index_string]
    for index in index_array:
        ax.annotate("%d" % (i+1), (data[:,0][index], expectation[index]), (data[:,0][index]-0.02, expectation[index]+0.01), fontsize=10)
        i += 1

ax.set_box_aspect(1)
plt.text(0.05, 0.95, "RMSE: {:.5f}".format(rmse), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.savefig("figures/%s/prediction_target_%d.pdf" % (net_type, run), bbox_inches="tight")
if show_plot == True:
    plt.show()
plt.close()

plt.plot(10*np.arange(train_loss.size), train_loss, label="Train Loss", color=soft_blue)
plt.plot(10*np.arange(val_loss.size), val_loss, label="Validation Loss", color=soft_red)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("figures/%s/loss_%d.pdf" % (net_type, run), bbox_inches="tight")
if show_plot == True:
    plt.show()
plt.close()
