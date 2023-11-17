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

# SHOW FIGURES?
show_figures = False

# bins in plots for std_dev
bins = 75

prediction_actual_base = np.load("./plotting_data/prediction_actual_base.npy")
loss_base = np.load("./plotting_data/loss_base.npy")
loss_base = loss_base[loss_base > 0.0001]

prediction_actual_cluster = np.load("./plotting_data/prediction_actual_cluster.npy")
loss_cluster = np.load("./plotting_data/loss_cluster.npy")
loss_cluster = loss_cluster[loss_cluster > 0.0001]

prediction_actual_cluster_perimeter = np.load("./plotting_data/prediction_actual_cluster_perimeter.npy")
loss_cluster_perimeter = np.load("./plotting_data/loss_cluster_perimeter.npy")
loss_cluster_perimeter = loss_cluster_perimeter[loss_cluster_perimeter > 0.0001]

label_data_base = torch.load("./training_data/label_data_base")
image_data_base = torch.load("./training_data/image_data_base")
label_data_cluster = torch.load("./training_data/label_data_cluster")
image_data_cluster = torch.load("./training_data/image_data_cluster")
label_data_cluster_perimeter = torch.load("./training_data/label_data_cluster_perimeter")
image_data_cluster_perimeter = torch.load("./training_data/image_data_cluster_perimeter")

label_data_base = label_data_base.numpy()
image_data_base = image_data_base.numpy()
label_data_cluster = label_data_cluster.numpy()
image_data_cluster = image_data_cluster.numpy()
label_data_cluster_perimeter = label_data_cluster_perimeter.numpy()
image_data_cluster_perimeter = image_data_cluster_perimeter.numpy()

image_data_base = image_data_base[label_data_base[:, -1].argsort()]
label_data_base = label_data_base[label_data_base[:, -1].argsort()]
image_data_cluster = image_data_cluster[label_data_cluster[:, -1].argsort()]
label_data_cluster = label_data_cluster[label_data_cluster[:, -1].argsort()]
image_data_cluster_perimeter = image_data_cluster_perimeter[label_data_cluster_perimeter[:, -1].argsort()]
label_data_cluster_perimeter = label_data_cluster_perimeter[label_data_cluster_perimeter[:, -1].argsort()]

# colors
soft_red = "#ed9688"
soft_blue = "#78b3eb"
soft_green = "#9befdf"

## left off here ##

# def fit
##########
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def sigmoid2d(X, L1, L2, L3, x0, y0, xy0, k1, k2, k3, b):
    x, y = X
    z = L1 / (1 + np.exp(-k1*(x-x0))) + L2 / (1 + np.exp(-k2*(y-y0))) + L3 / (1 + np.exp(-k3*(x*y-xy0))) + b
    return z
##########

# save fit parameters
#########
popt_sig_clust = np.load("./data_process/popt_sig_clust.npy")
popt_sig2d = np.load("./data_process/popt_sig2d.npy")
#########

# committor against cluster size
##########
plt.scatter(label_data_cluster[:,-1], label_data_cluster[:,0], s=1, c=soft_blue, label="Data")
plt.plot(label_data_cluster[:,-1], sigmoid(label_data_cluster[:,-1], *popt_sig_clust), c=soft_red, label="Sigmoid Fit")
plt.legend()
plt.title("Test Data Distribution")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Committor")
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig("./figures/data_dist_clust.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
##########

# base neural network prediction against actual
##########
# calculations
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_base[:,0])
min_val = min(prediction_actual_base[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_base[:,0] >= spacing[i], prediction_actual_base[:,0] < spacing[i+1]))
    
    std_dev[i] = np.std(prediction_actual_base[index][:,-1])
    mean[i] = np.mean(prediction_actual_base[index][:,-1])

std_dev_total = np.std(prediction_actual_base[:,-1]-prediction_actual_base[:,0])

line = np.linspace(min_val, max_val, bins+1)
line += (max_val-min_val)/(bins+1)
line = np.delete(line, -1)

# plotting
plt.plot(line, line, c=soft_red, label="Predicted = Actual")
plt.plot(line, mean, c=soft_blue, label="Mean Prediction")
plt.fill_between(line, mean - std_dev, mean + std_dev, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Base Neural Network Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Predicted")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "Standard Deviation: {}".format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/base_net_validation.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
#########

# cluster fit combined with correction neural network prediction
#########
# calculations
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_cluster[:,0])
min_val = min(prediction_actual_cluster[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_cluster[:,0] >= spacing[i], prediction_actual_cluster[:,0] < spacing[i+1]))
    combination = prediction_actual_cluster[index][:,-1]+prediction_actual_cluster[index][:,2]
    std_dev[i] = np.std(combination)
    mean[i] = np.mean(combination)

std_dev_total = np.std(prediction_actual_cluster[:,-1]+prediction_actual_cluster[:,2]-prediction_actual_cluster[:,0])

line = np.linspace(min_val, max_val, bins+1)
line += (max_val-min_val)/(bins+1)
line = np.delete(line, -1)

#plotting
plt.plot(line, line, c=soft_red, label="Predicted = Actual")
plt.plot(line, mean, c=soft_blue, label="Mean Prediction")
plt.fill_between(line, mean - std_dev, mean + std_dev, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("LCS Neural Network Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Predicted")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "Standard Deviation: {}".format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/cluster_net_validation_combination.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
#########

# LCS-P fit combined with correction neural network prediction
#########
# calculations
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_cluster_perimeter[:,0])
min_val = min(prediction_actual_cluster_perimeter[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_cluster_perimeter[:,0] >= spacing[i], prediction_actual_cluster_perimeter[:,0] < spacing[i+1]))
    combination = prediction_actual_cluster_perimeter[index][:,-1]+prediction_actual_cluster_perimeter[index][:,6]
    std_dev[i] = np.std(combination)
    mean[i] = np.mean(combination)

std_dev_total = np.std(prediction_actual_cluster_perimeter[:,-1]+prediction_actual_cluster_perimeter[:,6]-prediction_actual_cluster_perimeter[:,0])

line = np.linspace(min_val, max_val, bins+1)
line += (max_val-min_val)/(bins+1)
line = np.delete(line, -1)

#plotting
plt.plot(line, line, c=soft_red, label="Predicted = Actual")
plt.plot(line, mean, c=soft_blue, label="Mean Prediction")
plt.fill_between(line, mean - std_dev, mean + std_dev, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("LCS-P Neural Network Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Predicted")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "Standard Deviation: {}".format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/cluster_perimeter_net_validation_combination.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
#########

# max cluster fit prediction against actual
#########
# std dev for largest cluster fit
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_cluster[:,0])
min_val = min(prediction_actual_cluster[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_base[:,0] >= spacing[i], prediction_actual_base[:,0] < spacing[i+1]))
    std_dev[i] = np.std(sigmoid(prediction_actual_base[index][:,-2], *popt_sig_clust))
    mean[i] = np.mean(sigmoid(prediction_actual_base[index][:,-2], *popt_sig_clust))

std_dev_total = np.std(prediction_actual_base[:,0]-sigmoid(prediction_actual_base[:,-2], *popt_sig_clust))

line = np.linspace(min_val, max_val, bins+1)
line += (max_val-min_val)/(bins+1)
line = np.delete(line, -1)

# plotting
plt.plot(line, line, c=soft_red, label="Sigmoid Fit = Actual")
plt.plot(line, mean, c=soft_blue, label="Mean Sigmoid Fit")
plt.fill_between(line, mean - std_dev, mean + std_dev, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Largest Cluster Fit Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Fit")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "Standard Deviation: {}".format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/cluster_validation.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
##########

# cluster-perimeter fit prediction against actual
#########
# std dev for cluster-perimeter fit
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_cluster_perimeter[:,0])
min_val = min(prediction_actual_cluster_perimeter[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_cluster_perimeter[:,0] >= spacing[i], prediction_actual_cluster_perimeter[:,0] < spacing[i+1]))
    std_dev[i] = np.std(sigmoid2d([prediction_actual_cluster_perimeter[index][:,-2], prediction_actual_cluster_perimeter[index][:,-3]], *popt_sig2d))
    mean[i] = np.mean(sigmoid2d([prediction_actual_cluster_perimeter[index][:,-2], prediction_actual_cluster_perimeter[index][:,-3]], *popt_sig2d))

std_dev_total = np.std(prediction_actual_cluster_perimeter[:,0]-sigmoid2d([prediction_actual_cluster_perimeter[:,-2], prediction_actual_cluster_perimeter[:,-3]], *popt_sig2d))

line = np.linspace(min_val, max_val, bins+1)
line += (max_val-min_val)/(bins+1)
line = np.delete(line, -1)

#plotting
plt.plot(line, line, c=soft_red, label="Sigmoid Fit = Actual")
plt.plot(line, mean, c=soft_blue, label="Mean Sigmoid Fit")
plt.fill_between(line, mean - std_dev, mean + std_dev, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Cluster-Perimeter Fit Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Sigmoid Fit")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "Standard Deviation: {}".format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/sigmoid_2d_validation.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
#########


def f(x, A, B):
    return A*x + B

# residual of base network and lcs network
#########
# correlation coefficient
x = prediction_actual_base[:,0]-prediction_actual_base[:,-1]
y = prediction_actual_base[:,0]-sigmoid(prediction_actual_base[:,-2], *popt_sig_clust)

corr_coeff = np.corrcoef(x,y)[0,1]

popt, pcov = curve_fit(f, x, y)

plt.scatter(x, y, s=1, c=soft_blue, label="Data")
fit_x = np.array([min(x),max(x)])
plt.plot(fit_x,f(fit_x, *popt), c=soft_red, label="Straight Line Fit")
plt.title("Validation Residuals of Base Network and LCS Fit")
plt.xlabel("Network")
plt.ylabel("Fit")
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "Correlation Coefficient: {}".format(round(corr_coeff,3)), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/residuals_cluster.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
##########

# residual of network and cluster-perimeter fit
#########
# correlation coefficient
x = prediction_actual_base[:,0]-prediction_actual_base[:,-1]
y = prediction_actual_base[:,0]-sigmoid2d([prediction_actual_base[:,-2],prediction_actual_base[:,-3]], *popt_sig2d)

corr_coeff = np.corrcoef(x,y)[0,1]

popt, pcov = curve_fit(f, x, y)

plt.scatter(x, y, s=1, c=soft_blue, label="Data")
fit_x = np.array([min(x),max(x)])
plt.plot(fit_x,f(fit_x, *popt), c=soft_red, label="Straight Line Fit")
plt.title("Validation Residuals of Base Network and LCS-P Fit")
plt.xlabel("Network")
plt.ylabel("Fit")
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, "Correlation Coefficient: {}".format(round(corr_coeff,3)), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/residuals_cluster_perimeter.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
##########

# loss base
##########
plt.plot(loss_base, c=soft_blue)
plt.title("Mean Square Error (MSE) Loss During Training Of Base")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig("./figures/base_loss.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
##########

# loss lcs
##########
plt.plot(loss_cluster, c=soft_blue)
plt.title("Mean Square Error (MSE) Loss During Training Of LCS")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig("./figures/loss_cluster.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
##########

# loss lcs p
##########
plt.plot(loss_cluster_perimeter, c=soft_blue)
plt.title("Mean Square Error (MSE) Loss During Training Of LCS-P")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig("./figures/loss_cluster_perimeter.pdf", bbox_inches="tight")
if show_figures:
    plt.show()
plt.close()
##########
