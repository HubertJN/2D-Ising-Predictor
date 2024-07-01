import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 30
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams["figure.dpi"] = 512

if len(sys.argv) > 1: # if command line arguments provided
    net_type = sys.argv[1].lower()
    run = int(sys.argv[2])
else:
    print("Error. Specify run")
    exit()

data = np.load("./plotting_data/%s/%s_prediction_actual_%d.npy" % (net_type, net_type, run))
image_data_subset = torch.load("./training_data/image_data_subset").numpy()

num_select = 3

alpha = data[:,1]
beta = data[:,2]
expectation = alpha/(alpha+beta)
error = data[:,0]-expectation
over_index = np.argpartition(error, -num_select)[-num_select:]
under_index = np.argpartition(error, num_select)[:num_select]
ideal_index = np.argpartition(abs(error), num_select)[:num_select]

grid_index = data[over_index][:,3].astype(np.int32)
grid_over = np.squeeze(image_data_subset[grid_index])

grid_index = data[under_index][:,3].astype(np.int32)
grid_under = np.squeeze(image_data_subset[grid_index])

grid_index = data[ideal_index][:,3].astype(np.int32)
grid_ideal = np.squeeze(image_data_subset[grid_index])

rows, cols = grid_over[0].shape
cross_size = 0.5

grid_dict = {
    "over" : grid_over,
    "ideal" : grid_ideal,
    "under" : grid_under
}
index_dict = {
    "over" : over_index,
    "ideal" : ideal_index,
    "under" : under_index
}

l = 0
for type_string in ["over", "ideal", "under"]:
    grid = grid_dict[type_string]
    index = index_dict[type_string]
    for k in range(num_select):
        grid_sub = grid[k]
        index_sub = index[k]
        plt.figure()
    
        # Loop through each cell in the array
        for i in range(rows):
            for j in range(cols):
                if grid_sub[i, j] == 1:
                    # Plot a filled square with crosses
                    plt.fill([j, j, j+1, j+1], [i, i+1, i+1, i], 'w', edgecolor='black', lw=0.5)
                    plt.plot([j + cross_size/2, j + 1 - cross_size/2], [i + cross_size/2, i + 1 - cross_size/2], 'k', linewidth=0.5)
                    plt.plot([j + 1 - cross_size/2, j + cross_size/2], [i + cross_size/2, i + 1 - cross_size/2], 'k', linewidth=0.5) 
        
        plt.title("Grid %d" % (l+1))
        plt.xlabel("rmse: %f" % error[index_sub])
        plt.xlim(0, cols)
        plt.ylim(0, rows)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])
        plt.savefig("figures/%s/%s_%d_%d.pdf" % (net_type, type_string, k, run), bbox_inches="tight") 
        plt.close()
        l += 1

