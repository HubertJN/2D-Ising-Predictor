import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys
from scipy.interpolate import interp2d
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

rmse = np.array([
                [1,1,0.19810],
                [2,1,0.13430],
                [3,1,0.10870],
                [4,1,0.08743],
                [5,1,0.08177],
                [6,1,0.10628],
                [1,2,0.16255],
                [2,2,0.13022],
                [3,2,0.10264],
                [4,2,0.09650],
                [5,2,0.08782],
                [6,2,0.07609],
                [1,3,0.17676],
                [2,3,0.13649],
                [3,3,0.11385],
                [4,3,0.08390],
                [5,3,0.08761],
                [6,3,0.11520],
                [1,4,0.16400],
                [2,4,0.15958],
                [3,4,0.13950],
                [4,4,0.12763],
                [5,4,0.08392],
                [6,4,0.08184],
                [1,5,0.17403],
                [2,5,0.15956],
                [3,5,0.11321],
                [4,5,0.10756],
                [5,5,0.11049],
                [6,5,0.09503],
                [1,6,0.18323],
                [2,6,0.13596],
                [3,6,0.18967],
                [4,6,0.13143],
                [5,6,0.18671],
                [6,6,0.18944]
                ])

x_list = rmse[:,0]
y_list = rmse[:,1]
z_list = rmse[:,2]

idx = np.argmin(z_list)
print(rmse[idx])

Z = np.reshape(z_list, [6,6])
Z = Z[::-1]

fig = plt.imshow(Z,
                extent=[min(x_list),max(x_list),min(y_list),max(y_list)],
                cmap="viridis_r")
# Show the positions of the sample points, just to have some reference
fig.axes.set_autoscale_on(False)
plt.scatter(x_list,y_list,400,facecolors='none')
plt.xlabel("k_edge")
plt.ylabel("hidden_n")
plt.colorbar()
plt.show()
