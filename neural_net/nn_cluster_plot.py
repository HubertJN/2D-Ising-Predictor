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

data = torch.load("./training_data/label_data_subset").numpy()
image_data_subset = torch.load("./training_data/image_data_subset").numpy()
ordering = data[:,1].argsort()
image_data_subset = image_data_subset[ordering]
data = data[ordering]
committor = data[:,0]
geo_clust = data[:,1]

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

p0 = [max(committor), np.median(geo_clust),1,min(committor)]
popt, pcov = curve_fit(sigmoid, geo_clust, committor, p0=p0)

plt.plot(geo_clust, sigmoid(geo_clust, *popt), c=soft_red)
plt.scatter(geo_clust, committor, c=soft_blue, s=0.5)
plt.ylabel('Committor')
plt.xlabel('Geometric Cluster Size')
plt.tight_layout()
plt.savefig("./figures/cluster_fit.pdf", bbox_inches="tight")
plt.close()

line = np.linspace(np.min(committor),np.max(committor),10)
rmse = np.sqrt(np.mean((committor-sigmoid(geo_clust, *popt))**2))

plt.plot(line, line, color=soft_red)
plt.scatter(committor, sigmoid(geo_clust, *popt), s=1, color=soft_blue)
plt.xlabel("Target")
plt.ylabel("Prediction")
ax = plt.gca()
plt.text(0.05, 0.95, "RMSE: {:.5f}".format(rmse), transform = ax.transAxes, horizontalalignment="left",
     verticalalignment="top")
plt.tight_layout()
plt.savefig("./figures/prediction_actual_cluster.pdf", bbox_inches="tight")
plt.close()