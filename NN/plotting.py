import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams['figure.dpi'] = 120

prediction_actual = np.load("prediction_actual.npy")
loss = np.load("loss.npy")
loss = loss[loss > 0.001]

committor_data = torch.load("./committor_data")
grid_data = torch.load("./grid_data")

committor_data = committor_data.numpy()
grid_data = grid_data.numpy()
grid_data = grid_data[committor_data[:, -1].argsort()]
committor_data = committor_data[committor_data[:, -1].argsort()]

std_dev_net = np.zeros(50)
mean_net = np.zeros(50)
max_val = max(prediction_actual[:,2])
min_val = min(prediction_actual[:,2])
spacing = np.linspace(min_val, max_val, 51)
for i in range(50):
    index = np.where(np.logical_and(prediction_actual[:,2] >= spacing[i], prediction_actual[:,2] < spacing[i+1]))
    
    std_dev_net[i] = np.std(prediction_actual[index][:,-1])
    mean_net[i] = np.mean(prediction_actual[index][:,-1])

std_dev_net_total = np.std(prediction_actual[:,-1]-prediction_actual[:,2])

# colors
soft_red = "#ed9688"
soft_blue = "#78b3eb"
soft_green = "#9befdf"

# Ising grid
cluster_choice = 200
for grid_choice in [6]:
    grid_index = np.where(committor_data[:,-1] == cluster_choice)
    plt.pcolormesh(grid_data[grid_index][grid_choice,0], cmap='Greys_r', edgecolors='k', linewidth=0.1)
    plt.title("Ising Grid", y=1.01)
    #plt.title("{}".format(grid_choice))
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig("figures/ising_grid.pdf", bbox_inches='tight')
    plt.show()
    plt.close()

print("Committor, committor std dev and largest cluster size for saved ising grid:", committor_data[grid_index][grid_choice,0], committor_data[grid_index][grid_choice,1], cluster_choice)

# def fit
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
p0 = [max(committor_data[:,0]), np.median(committor_data[:,-1]),1,min(committor_data[:,0])]

popt, _ = curve_fit(sigmoid, committor_data[:,-1], committor_data[:,0], p0=p0)

# committor against cluster size
plt.scatter(committor_data[:,-1], committor_data[:,0], s=1, c=soft_blue, label="Data")
plt.plot(committor_data[:,-1], sigmoid(committor_data[:,-1], *popt), c=soft_red, label="Sigmoid Fit")
plt.legend()
plt.title("Test Data Distribution")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Committor")
plt.legend(loc="lower right")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('figures/data_dist.pdf', bbox_inches='tight')
plt.show()
plt.close()

# neural network prediction against actual
#plt.scatter(prediction_actual[:,0],prediction_actual[:,-1], s=1, c=soft_red, label="Data")
line = np.linspace(min_val, max_val, 51)
line += (max_val-min_val)/51
line = np.delete(line, -1)

plt.plot(line, line, c=soft_red, label="Predicted = Actual")
plt.plot(line, mean_net, c=soft_blue, label="Mean Prediction")
plt.fill_between(line, mean_net - std_dev_net, mean_net + std_dev_net, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Neural Network Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Predicted")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_net_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/net_validation.pdf', bbox_inches='tight')
plt.show()
plt.close()

# neural network prediction including sigmoid fit
# calculations
line = np.linspace(0, 1, 101)
std_dev_net_combined = np.zeros(101)
mean_net_combined = np.zeros(101)
for i in range(101):
    index_combined = np.where((np.round(100*prediction_actual[:,0])).astype(np.int32) == i)
    combination = prediction_actual[index_combined][:,-1]+prediction_actual[index_combined][:,1]
    std_dev_net_combined[i] = np.std(combination)
    mean_net_combined[i] = np.mean(combination)

std_dev_net_total_combined = np.std(prediction_actual[:,-1]+prediction_actual[:,1]-prediction_actual[:,0])

#plotting
plt.plot(line, line, c=soft_red, label="Predicted = Actual")
plt.plot(line, mean_net_combined, c=soft_blue, label="Mean Prediction")
plt.fill_between(line, mean_net_combined - std_dev_net_combined, mean_net_combined + std_dev_net_combined, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Combined Neural Network Prediction Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Predicted")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_net_total_combined,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/net_validation_combination.pdf', bbox_inches='tight')
plt.show()
plt.close()


# std dev for fit
std_dev_fit = np.zeros(101)
mean_fit = np.zeros(101)
for i in range(101):
    index = np.where((np.round(100*prediction_actual[:,0])).astype(np.int32) == i)
    std_dev_fit[i] = np.std(sigmoid(prediction_actual[index][:,-2], *popt))
    mean_fit[i] = np.mean(sigmoid(prediction_actual[index][:,-2], *popt))

std_dev_fit_total = np.std(prediction_actual[:,0]-sigmoid(prediction_actual[:,-2], *popt))

# max cluster fit prediction against actual
line = np.linspace(0, 1, 101)
plt.plot(line, line, c=soft_red, label="Sigmoid Fit = Actual")
plt.plot(line, mean_fit, c=soft_blue, label="Mean Sigmoid Fit")
plt.fill_between(line, mean_fit - std_dev_fit, mean_fit + std_dev_fit, alpha=0.2, step="mid", color=soft_blue, label="Prediction Standard Deviation")
plt.legend(loc="lower right")
plt.title("Sigmoid Fit Validation")
plt.xlabel("Actual")
plt.ylabel("Mean Sigmoid Fit")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, 'Standard Deviation: {}'.format(round(std_dev_fit_total,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/sigmoid_validation.pdf', bbox_inches='tight')
plt.show()
plt.close()

# correlation coefficient
x = prediction_actual[:,0]-(prediction_actual[:,-1]+prediction_actual[:,1])
y = prediction_actual[:,0]-sigmoid(prediction_actual[:,-2], *popt)

corr_coeff = np.corrcoef(x,y)[0,1]

# residual of network and fit
plt.scatter(x, y, s=1, c=soft_blue, label="Data")
line = np.linspace(0, 1, 101)
plt.title("Validation Residuals")
plt.xlabel("Network")
plt.ylabel("Fit")
ax = plt.gca()
ax.set_box_aspect(1)
plt.text(0.05, 0.95, 'Correlation Coefficient: {}'.format(round(corr_coeff,3)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top')
plt.tight_layout()
plt.savefig('figures/residuals.pdf', bbox_inches='tight')
plt.show()
plt.close()

# loss
plt.plot(loss, c=soft_blue)
plt.title("Mean Square Error (MSE) Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
ax = plt.gca()
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('figures/loss.pdf', bbox_inches='tight')
plt.show()
plt.close()
