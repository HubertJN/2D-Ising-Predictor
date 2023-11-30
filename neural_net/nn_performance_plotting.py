import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams["figure.dpi"] = 120

#filenames_channels = []
x = np.linspace(1,1000,11,dtype=np.int32)
#for i in np.linspace(1,64,17,dtype=np.int32):
#    filenames_channels.append("".join(["./performance/loss_base_", str(2*(i+1)), "_1.npy"]))

#channels_arr = np.array([np.load(fname) for fname in filenames_channels], dtype=object)

#channel_mins = np.array([arr.min() for arr in channels_arr])

#plt.plot(x,channel_mins)
#plt.show()

test_loss_parameters = np.load("./performance/test_loss_parameters.npy")

print(int(test_loss_parameters[test_loss_parameters[:,2].astype(np.int32) == 100][np.argmin(test_loss_parameters[test_loss_parameters[:,2].astype(np.int32) == 100][:,0]),1]))

cmap = plt.cm.coolwarm
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cmap(np.linspace(0, 1, 17)))

num_channels = 1
for i in np.linspace(1, 64, 17, dtype=np.int32):
    y = test_loss_parameters[test_loss_parameters[:,1].astype(np.int32) == i]
    if i == 1 or i == 32 or i == 64:
        plt.plot(x,y[:,0], label="Channels = {}".format(i))
    else:
        plt.plot(x,y[:,0])
    plt.ylabel("Test Loss")
    plt.xlabel("Linear Layer Nodes")

plt.legend()
plt.title("Performance of NN with respect to Number of Parameters")
#plt.xlim(0,500)
#plt.ylim(0.0, 0.005)
plt.yscale("log")
plt.show()

plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']

test_loss_parameters = np.load("./performance/test_loss_dataset.npy")

# exp fit
def exp(x, a, b, c):
    return a * x**(-b) + c
    #return a * np.exp(-b * x) + c

p0_exp = [1, 1, 0]
popt_exp, _ = curve_fit(exp, test_loss_parameters[:,1], test_loss_parameters[:,0], p0=p0_exp)

plt.plot(test_loss_parameters[:,1], test_loss_parameters[:,0], label="Data")
plt.plot(np.linspace(1,20000, 1000), exp(np.linspace(1,20000, 1000), *popt_exp), label="Fit")
plt.title("Performance of NN with respect to size of Training Set")
plt.ylabel("Test Loss")
plt.xlabel("Dataset")

#plt.xlim(0,50)
plt.ylim(0, 0.015)
plt.legend()
plt.show()

exit()

filenames_linear_nodes = []
for i in range(11):
    filenames_linear_nodes.append("".join(["./performance/loss_base_1_", str(2**i), ".npy"]))

linear_nodes_arr = np.array([np.load(fname) for fname in filenames_linear_nodes])

linear_nodes_mins = linear_nodes_arr.min(axis=0)

plt.plot(linear_nodes_mins)
plt.show()