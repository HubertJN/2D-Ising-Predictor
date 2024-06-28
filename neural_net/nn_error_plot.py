import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams["figure.dpi"] = 120

if len(sys.argv) > 1: # if command line arguments provided
    net_type = sys.argv[1].lower()
    run = int(sys.argv[2])
else:
    print("Error. Specify run")
    exit()

data = np.load("./plotting_data/%s/%s_prediction_actual_%d.npy" % (net_type, net_type, run))
image_data_subset = torch.load("./training_data/image_data_subset").numpy()

num_top = 5

alpha = data[:,1]
beta = data[:,2]
expectation = alpha/(alpha+beta)
error = abs(data[:,0]-expectation)
top_max = np.argpartition(error, -num_top)[-num_top:]
grid_index = data[top_max][:,3].astype(np.int32)
grid_info = np.squeeze(image_data_subset[grid_index])
rows, cols = grid_info[0].shape
cross_size = 0.5

for grid in range(num_top):
    grid_sub = grid_info[grid]
    plt.figure()
 
    # Loop through each cell in the array
    for i in range(rows):
        for j in range(cols):
            if grid_sub[i, j] == 1:
                # Plot a filled square with crosses
                plt.fill([j, j, j+1, j+1], [i, i+1, i+1, i], 'w', edgecolor='black', lw=0.5)
                plt.plot([j + cross_size/2, j + 1 - cross_size/2], [i + cross_size/2, i + 1 - cross_size/2], 'k', linewidth=0.5)
                plt.plot([j + 1 - cross_size/2, j + cross_size/2], [i + cross_size/2, i + 1 - cross_size/2], 'k', linewidth=0.5) 
    
    plt.title("Grid %d" % (grid+1))
    #plt.xlabel("Grid index: %d" % top_max[grid])
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("figures/%s/error_%d_%d.pdf" % (net_type, grid, run), bbox_inches="tight") 
    plt.close()
