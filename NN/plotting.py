import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Times New Roman"

prediction_actual = np.load("prediction_actual.npy")
loss = np.load("loss.npy")
loss = loss[loss > 0.001]

committor_data = torch.load("./committor_data_tmp")

committor_data = committor_data.numpy()
committor_data = committor_data[committor_data[:, 2].argsort()]

std_dev = np.zeros(101)
comm_percent = 0
for i in range(101):
    index = np.where((np.round(100*prediction_actual[:,0])).astype(np.int32) == i)
    std_dev[i] = np.std(prediction_actual[index][:,-1])
    comm_percent += np.sum((abs(prediction_actual[index][:,-1]-prediction_actual[index][:,0])) < std_dev[i])

comm_percent = comm_percent/len(prediction_actual)*100

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
plt.plot(committor_data[:,2], sigmoid(committor_data[:,2], *popt), c=soft_red, label="Fit")
plt.legend()
plt.title("Test Data Distribution")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Committor")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# neural network prediction against actual
plt.scatter(prediction_actual[:,0],prediction_actual[:,-1], s=1, c=soft_red, label="Data")
line = np.linspace(0, 1, 101)
plt.plot(line, line, c=soft_blue, label="Predicted = Actual")
plt.fill_between(line, line - std_dev, line + std_dev, alpha=0.4, step="mid", color=soft_blue, label="Standard Deviation")
plt.legend(loc="lower right")
plt.title("Neural Network Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.text(0.01, 0.99, 'Predictions within std dev: {}%'.format(round(comm_percent,1)))
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# std dev for fit
for i in range(101):
    index = np.where((np.round(100*prediction_actual[:,0])).astype(np.int32) == i)
    std_dev[i] = np.std(sigmoid(prediction_actual[index][:,2], *popt))
    comm_percent += np.sum((abs(sigmoid(prediction_actual[index][:,2], *popt)-prediction_actual[index][:,0])) < std_dev[i])

comm_percent = comm_percent/len(prediction_actual)*100

# max cluster fit prediction against actual
plt.scatter(prediction_actual[:,0], sigmoid(prediction_actual[:,2], *popt), s=1, c=soft_red, label="Data")
line = np.linspace(0, 1, 101)
plt.plot(line, line, c=soft_blue, label="Fit = Actual")
plt.fill_between(line, line - std_dev, line + std_dev, alpha=0.4, step="mid", color=soft_blue, label="Standard Deviation")
plt.legend(loc="lower right")
plt.title("Fit Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.text(0.01, 0.99, 'Predictions within std dev: {}%'.format(round(comm_percent,1)))
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
