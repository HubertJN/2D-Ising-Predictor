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
plt.rcParams['figure.dpi'] = 120

# SHOW FIGURES?
show_figures = True

# bins in plots for std_dev
bins = 75

prediction_actual_base = np.load("prediction_actual_base.npy")
loss_base = np.load("loss_base.npy")
loss_base = loss_base[loss_base > 0.0001]

prediction_actual_lcs = np.load("prediction_actual_lcs.npy")
loss_lcs = np.load("loss_lcs.npy")
loss_lcs = loss_lcs[loss_lcs > 0.0001]

prediction_actual_lcs_p = np.load("prediction_actual_lcs_p.npy")
loss_lcs_p = np.load("loss_lcs_p.npy")
loss_lcs_p = loss_lcs_p[loss_lcs_p > 0.0001]

committor_data = torch.load("./committor_data")
grid_data = torch.load("./grid_data")

committor_data = committor_data.numpy()
grid_data = grid_data.numpy()
grid_data = grid_data[committor_data[:, -1].argsort()]
committor_data = committor_data[committor_data[:, -1].argsort()]

# colors
soft_red = "#ed9688"
soft_blue = "#78b3eb"
soft_green = "#9befdf"

# Ising grid
cluster_choice = 200
for grid_choice in [6]:
    grid_index = np.where(committor_data[:,-1] == cluster_choice)
    grid_info = grid_data[grid_index][grid_choice,0]
    plt.pcolormesh(grid_info, cmap='Greys_r', edgecolors='k', linewidth=0.1)
    plt.title("Ising Grid", y=1.01)
    #plt.title("{}".format(grid_choice))
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig("figures/ising_grid.pdf", bbox_inches='tight')
    if show_figures:
        plt.show()
    plt.close()

# def fit
##########
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
p0_sig_clust = [max(committor_data[:,0]), np.median(committor_data[:,-1]),1,min(committor_data[:,0])]
p0_sig_perim = [max(committor_data[:,0]), np.median(committor_data[:,-2]),1,min(committor_data[:,0])]

def sigmoid2d(X, L1, L2, x0, y0, k1, k2, b):
    x, y = X
    z = L1 / (1 + np.exp(-k1*(x-x0))) * L2 / (1 + np.exp(-k2*(y-y0))) + b
    return z
p0_sig2d = [max(committor_data[:,0]), max(committor_data[:,0]), np.median(committor_data[:,-1]), np.median(committor_data[:,-2]), 1, 1, min(committor_data[:,0])]

popt_sig_clust, _ = curve_fit(sigmoid, committor_data[:,-1], committor_data[:,0], p0=p0_sig_clust)
popt_sig_perim, _ = curve_fit(sigmoid, committor_data[:,-2], committor_data[:,0], p0=p0_sig_perim)
popt_sig2d, _ = curve_fit(sigmoid2d, [committor_data[:,-1], committor_data[:,-2]], committor_data[:,0], p0=p0_sig2d)
##########

# committor against cluster size
##########
plt.scatter(committor_data[:,-1], committor_data[:,0], s=1, c=soft_blue, label="Data")
plt.plot(committor_data[:,-1], sigmoid(committor_data[:,-1], *popt_sig_clust), c=soft_red, label="Sigmoid Fit")
plt.legend()
plt.title("Test Data Distribution")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Committor")
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('figures/data_dist_clust.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
##########

# committor against perimeter
##########
plt.scatter(committor_data[:,-2], committor_data[:,0], s=1, c=soft_blue, label="Data")
perimeter_plot_y = sigmoid(committor_data[:,-2], *popt_sig_perim)
perimeter_plot_sort = perimeter_plot_y.argsort()
plt.plot(committor_data[perimeter_plot_sort][:,-2], perimeter_plot_y[perimeter_plot_sort], c=soft_red, label="Sigmoid Fit")
plt.legend()
plt.title("Test Data Distribution")
plt.xlabel("Perimeter")
plt.ylabel("Committor")
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('figures/data_dist_perim.pdf', bbox_inches='tight')
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
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/base_net_validation.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
#########

# LCS fit combined with correction neural network prediction
#########
# calculations
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_lcs[:,0])
min_val = min(prediction_actual_lcs[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_lcs[:,0] >= spacing[i], prediction_actual_lcs[:,0] < spacing[i+1]))
    combination = prediction_actual_lcs[index][:,-1]+prediction_actual_lcs[index][:,2]
    std_dev[i] = np.std(combination)
    mean[i] = np.mean(combination)

std_dev_total = np.std(prediction_actual_lcs[:,-1]+prediction_actual_lcs[:,2]-prediction_actual_lcs[:,0])

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
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/lcs_net_validation_combination.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
#########

# LCS-P fit combined with correction neural network prediction
#########
# calculations
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_lcs_p[:,0])
min_val = min(prediction_actual_lcs_p[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_lcs_p[:,0] >= spacing[i], prediction_actual_lcs_p[:,0] < spacing[i+1]))
    combination = prediction_actual_lcs_p[index][:,-1]+prediction_actual_lcs_p[index][:,6]
    std_dev[i] = np.std(combination)
    mean[i] = np.mean(combination)

std_dev_total = np.std(prediction_actual_lcs_p[:,-1]+prediction_actual_lcs_p[:,6]-prediction_actual_lcs_p[:,0])

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
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/lcs_p_net_validation_combination.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
#########

# max cluster fit prediction against actual
#########
# std dev for largest cluster fit
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_lcs[:,0])
min_val = min(prediction_actual_lcs[:,0])
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
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/cluster_validation.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
##########

# perimeter fit prediction against actual
#########
# std dev for perimeter fit
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_lcs_p[:,0])
min_val = min(prediction_actual_lcs_p[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_base[:,0] >= spacing[i], prediction_actual_base[:,0] < spacing[i+1]))
    std_dev[i] = np.std(sigmoid(prediction_actual_base[index][:,-3], *popt_sig_perim))
    mean[i] = np.mean(sigmoid(prediction_actual_base[index][:,-3], *popt_sig_perim))

std_dev_total = np.std(prediction_actual_base[:,0]-sigmoid(prediction_actual_base[:,-3], *popt_sig_perim))

line = np.linspace(min_val, max_val, bins+1)
line += (max_val-min_val)/(bins+1)
line = np.delete(line, -1)

# plotting
plt.plot(line, line, c=soft_red, label="Sigmoid Fit = Actual")
plt.plot(line, mean, c=soft_blue, label="Mean Sigmoid Fit")
plt.fill_between(line, mean - std_dev, mean + std_dev, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Perimeter Fit Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Fit")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/perimeter_validation.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
##########

# cluster-perimeter fit prediction against actual
#########
# std dev for cluster-perimeter fit
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(prediction_actual_lcs_p[:,0])
min_val = min(prediction_actual_lcs_p[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(prediction_actual_base[:,0] >= spacing[i], prediction_actual_base[:,0] < spacing[i+1]))
    std_dev[i] = np.std(sigmoid2d([prediction_actual_lcs_p[index][:,-2], prediction_actual_lcs_p[index][:,-3]], *popt_sig2d))
    mean[i] = np.mean(sigmoid2d([prediction_actual_lcs_p[index][:,-2], prediction_actual_lcs_p[index][:,-3]], *popt_sig2d))

std_dev_total = np.std(prediction_actual_lcs_p[:,0]-sigmoid2d([prediction_actual_lcs_p[:,-2], prediction_actual_lcs_p[:,-3]], *popt_sig2d))

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
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/sigmoid_2d_validation.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
#########


def f(x, A, B):
    return A*x + B

# residual of base network and cluster fit
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
plt.text(0.05, 0.95, 'Correlation Coefficient: {}'.format(round(corr_coeff,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/lcs_residuals.pdf', bbox_inches='tight')
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
plt.text(0.05, 0.95, 'Correlation Coefficient: {}'.format(round(corr_coeff,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/lcs_p_residuals.pdf', bbox_inches='tight')
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
plt.savefig('figures/base_loss.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
##########

# loss lcs
##########
plt.plot(loss_lcs, c=soft_blue)
plt.title("Mean Square Error (MSE) Loss During Training Of LCS")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('figures/lcs_loss.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
##########

# loss lcs p
##########
plt.plot(loss_lcs_p, c=soft_blue)
plt.title("Mean Square Error (MSE) Loss During Training Of LCS-P")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('figures/lcs_p_loss.pdf', bbox_inches='tight')
if show_figures:
    plt.show()
plt.close()
##########
