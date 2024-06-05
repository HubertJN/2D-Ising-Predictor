import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

rmse = np.load("figures/gnn_tweaking/rmse.npy")
rmse_new = np.zeros([int(np.max(rmse[:,0])*np.max(rmse[:,1])),3])

# loops through rmse finding entries for same parameters and takes minimum loss
for i in range(0, int(np.max(rmse[:,0]))):
    for j in range(0, int(np.max(rmse[:,1]))):
        filtered = rmse[rmse[:,0] == i+1]
        filtered_array = filtered[filtered[:,1] == j+1]
        rmse_new[i*int(np.max(rmse[:,1]))+j] = filtered_array[np.argmin(filtered_array[:,2])]
        #rmse_new[i*int(np.max(rmse[:,1]))+j] = np.mean(filtered_array, axis=0)

rmse = rmse_new

x_list = rmse[:,0]
y_list = rmse[:,1]
z_list = rmse[:,2]

Z = np.reshape(z_list, [int(np.max(rmse[:,0])),int(np.max(rmse[:,1]))])
Z = Z[::-1]

fig = plt.imshow(Z,
                extent=[min(x_list)-0.5,max(x_list)+0.5,min(y_list)-0.5,max(y_list)+0.5],
                cmap="viridis_r")
# Show the positions of the sample points, just to have some reference
fig.axes.set_autoscale_on(False)
plt.scatter(x_list,y_list,400,facecolors='none')
plt.xlabel("k_edge")
plt.ylabel("hidden_n")
plt.colorbar()
plt.show()

