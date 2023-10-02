import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
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

committor_data = np.load("prediction_actual_lcs_p.npy")
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

grid_info = grid_data[:,0]
num_clust = np.zeros(grid_info.shape[0])

for i in range(len(num_clust)):
    clusters, num = ndimage.label(grid_info[i])
    area = ndimage.sum(grid_info[i], clusters, index=np.arange(clusters.max() + 1))
    step_one = clusters[clusters < np.argmax(area)]
    step_two = step_one[step_one != 0]
    num_clust[i] = len(step_two)

secondary = num_clust

# def fit
##########
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y
p0_sig_clust = [max(committor_data[:,0]), np.median(committor_data[:,-1]),1,min(committor_data[:,0])]
p0_sig_perim = [max(committor_data[:,0]), np.median(committor_data[:,-2]),1,min(committor_data[:,0])]

def sigmoid2d(X, L1, L2, L3, x0, y0, xy0, k1, k2, k3, b):
    x, y = X
    z = L1 / (1 + np.exp(-k1*(x-x0))) + L2 / (1 + np.exp(-k2*(y-y0))) + L3 / (1 + np.exp(-k3*(x*y-xy0))) + b
    return z
p0_sig2d = [max(committor_data[:,0]), max(committor_data[:,0]), max(committor_data[:,0]), np.median(committor_data[:,-1]), \
np.median(secondary), np.median(secondary*committor_data[:,-1]), 1, 1, 1, min(committor_data[:,0])]

popt_sig_clust, _ = curve_fit(sigmoid, committor_data[:,-1], committor_data[:,0], p0=p0_sig_clust)
popt_sig_perim, _ = curve_fit(sigmoid, committor_data[:,-2], committor_data[:,0], p0=p0_sig_perim)
popt_sig2d, _ = curve_fit(sigmoid2d, [committor_data[:,-1], secondary], committor_data[:,0], p0=p0_sig2d)

##########
# committor perimeter and cluster
#########
fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(np.min(prediction_actual_base[:,-2]), np.max(prediction_actual_base[:,-2]), 50)
y = np.linspace(np.min(secondary), np.max(secondary), 50)

X, Y = np.meshgrid(x, y)

Z = sigmoid2d([X, Y], *popt_sig2d)

#print(np.sum(grid_data, axis=(-1,-2))[:,0].shape)
length = len(committor_data[:,-1])
index = np.random.randint(0, length, 2000)
ax.scatter3D(committor_data[:,-1][index], secondary[index], committor_data[:,0][index], color=soft_red)
ax.plot_surface(X, Y, Z, color=soft_blue, alpha=0.5)
plt.show()
plt.close()
#########

# cluster-perimeter fit prediction against actual
#########
# std dev for cluster-perimeter fit
bins = 75
std_dev = np.zeros(bins)
mean = np.zeros(bins)
max_val = max(committor_data[:,0])
min_val = min(committor_data[:,0])
spacing = np.linspace(min_val, max_val, bins+1)
for i in range(bins):
    index = np.where(np.logical_and(committor_data[:,0] >= spacing[i], committor_data[:,0] < spacing[i+1]))
    std_dev[i] = np.std(sigmoid2d([committor_data[index][:,-1], secondary[index]], *popt_sig2d))
    mean[i] = np.mean(sigmoid2d([committor_data[index][:,-1], secondary[index]], *popt_sig2d))

std_dev_total = np.std(committor_data[:,0]-sigmoid2d([committor_data[:,-1], secondary], *popt_sig2d))

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