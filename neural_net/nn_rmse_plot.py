import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

if len(sys.argv) > 1:
    net_type = sys.argv[1].lower()
else:
    print("Error. No network type. Exiting.")

rmse = np.load("figures/%s/rmse.npy" % net_type)

x_list = rmse[:,0]
y_list = rmse[:,1]
z_list = rmse[:,2]


Z = np.reshape(z_list, [int(np.max(x_list)),int(np.max(y_list))]).T[::-1]
Z_ratio = Z.shape[0]/Z.shape[1]

fig = plt.imshow(Z,
                extent=[min(x_list)-0.5,max(x_list)+0.5,min(y_list)-0.5,max(y_list)+0.5],
                cmap="viridis_r")

ax = plt.gca()
ax.yaxis.get_major_locator().set_params(integer=True)
ax.xaxis.get_major_locator().set_params(integer=True)
plt.xlabel("k_edge")
plt.ylabel("hidden_n")
plt.colorbar(fig,fraction=0.047*Z_ratio, pad=0.04)
plt.savefig("figures/%s/rmse_hyperparameters.pdf" % net_type, bbox_inches="tight")

