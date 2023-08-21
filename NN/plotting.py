import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

prediction_actual = np.load("prediction_actual.npy")
loss = np.load("loss.npy")
loss = loss[loss > 0.001]

committor_data = torch.load("./committor_data_tmp")

committor_data = committor_data.numpy()
committor_data = committor_data[committor_data[:, 2].argsort()]

std_dev_net = np.zeros(101)
mean_net = np.zeros(101)
for i in range(101):
    index = np.where((np.round(100*prediction_actual[:,0])).astype(np.int32) == i)
    std_dev_net[i] = np.std(prediction_actual[index][:,-1])
    mean_net[i] = np.mean(prediction_actual[index][:,-1])

std_dev_net_total = np.std(prediction_actual[:,-1]-prediction_actual[:,0])

# colors
soft_red = "#ed9688"
soft_blue = "#78b3eb"
soft_green = "#9befdf"

# def fit
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
p0 = [max(committor_data[:,0]), np.median(committor_data[:,2]),1,min(committor_data[:,0])]

popt, _ = curve_fit(sigmoid, committor_data[:,2], committor_data[:,0], p0=p0)

# committor against cluster size
plt.scatter(committor_data[:,2], committor_data[:,0], s=1, c=soft_blue, label="Data")
plt.plot(committor_data[:,2], sigmoid(committor_data[:,2], *popt), c=soft_red, label="Sigmoid Fit")
plt.legend()
plt.title("Test Data Distribution")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Committor")
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# neural network prediction against actual
#plt.scatter(prediction_actual[:,0],prediction_actual[:,-1], s=1, c=soft_red, label="Data")
line = np.linspace(0, 1, 101)
plt.plot(line, line, c=soft_red, label="Predicted = Actual")
plt.plot(line, mean_net, c=soft_blue, label="Mean Prediction")
plt.fill_between(line, mean_net - std_dev_net, mean_net + std_dev_net, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Neural Network Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Predicted")
plt.text(0.01, 0.99, 'Standard Deviation: {}'.format(round(std_dev_net_total,3)))
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# std dev for fit
std_dev_fit = np.zeros(101)
mean_fit = np.zeros(101)
for i in range(101):
    index = np.where((np.round(100*prediction_actual[:,0])).astype(np.int32) == i)
    std_dev_fit[i] = np.std(sigmoid(prediction_actual[index][:,2], *popt))
    mean_fit[i] = np.mean(sigmoid(prediction_actual[index][:,2], *popt))

std_dev_fit_total = np.std(prediction_actual[:,0]-sigmoid(prediction_actual[:,2], *popt))

# max cluster fit prediction against actual
line = np.linspace(0, 1, 101)
plt.plot(line, line, c=soft_red, label="Sigmoid Fit = Actual")
plt.plot(line, mean_fit, c=soft_blue, label="Mean Sigoid Fit")
plt.fill_between(line, mean_fit - std_dev_fit, mean_fit + std_dev_fit, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Sigmoid Fit Validation")
plt.xlabel("Actual")
plt.ylabel("Sigmoid Fit")
plt.text(0.01, 0.99, 'Standard Deviation: {}'.format(round(std_dev_fit_total,3)))
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# residual of network and fit
plt.scatter(prediction_actual[:,0]-prediction_actual[:,-1], prediction_actual[:,0]-sigmoid(prediction_actual[:,2], *popt), s=1, c=soft_red, label="Data")
line = np.linspace(0, 1, 101)
plt.title("Residual")
plt.xlabel("Network")
plt.ylabel("Fit")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# loss
plt.plot(loss, c=soft_blue)
plt.title("Mean Square Error Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()
