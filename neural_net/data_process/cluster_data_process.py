import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.optimize import curve_fit

# parameters
gridstates = open("../../bin/gridstates.bin", "rb")
index_file = open("../../bin/committor_index.bin", "rb")
byte_prefix = 4*(3)
ngrids = 8192
lx = 64
ly = 64
ld_arr_size = 5

# count how many samples
i = 0
while(1):
    index = np.fromfile(index_file, dtype=np.int32, count=3, sep="")
    if index.size < 3:
        break
    committor = np.fromfile(index_file, dtype=np.float64, count=1, sep="")[0]
    std_dev = np.fromfile(index_file, dtype=np.float64, count=1, sep="")[0]
    i += 1

print("Number of initial samples: ", i)

index_file.seek(0, os.SEEK_SET)

bytes_per_slice = byte_prefix+ngrids*(lx*ly/8)
image_data = np.zeros((i,lx*ly), dtype=np.float32)
label_data = np.zeros((i, ld_arr_size), dtype=np.float32)
cluster_data = np.zeros(i, dtype=np.float32)
j = 0
ibit=0
ibyte=0
one = np.uint32(1)
blookup = [0, 1]

while(1):
    # index[0] - sweep, index[1] - ngrid, index[2] - cluster
    index = np.fromfile(index_file, dtype=np.int32, count=3, sep="")
    if index.size < 3:
        break
    index[0] = index[0]/100
    committor = np.fromfile(index_file, dtype=np.float64, count=1, sep="")[0]
    std_dev = np.fromfile(index_file, dtype=np.float64, count=1, sep="")[0]
    gridstates.seek(int(byte_prefix+bytes_per_slice*index[0]+(lx*ly/8)*(index[1])), os.SEEK_SET)
    bitgrid = np.fromfile(gridstates, dtype=np.byte, count=lx*ly, sep="")
    isite=0
    ising_grids = list(range(lx*ly))
    for ibyte in range(int(lx*ly/8)):
        for ibit in range(8):
            ising_grids[isite] = blookup[(bitgrid[ibyte] >> ibit) & one]
            isite += 1
            if (isite>lx*ly):
                break
    grid_info = np.reshape(np.array(ising_grids),[64,64])
    image_data[j] = ising_grids
    label_data[j, 0] = 0.0 # label
    label_data[j, 1] = 0.54 # inverse temperature $$(tmp)$$
    label_data[j, 2] = 0.07 # field strength $$(tmp)$$
    label_data[j, 3] = float(index[2]) # cluster size
    label_data[j, 4] = committor # committor
    cluster_data[j] = float(index[2])
    j += 1

try:
    os.remove("../training_data/image_data_cluster")
    os.remove("../training_data/label_data_cluster")
except:
    pass

# fix data
fix_index = (label_data[:, -1] >= 0) & (label_data[:, -1] <= 0.99)
image_data = image_data[fix_index]
label_data = label_data[fix_index]
cluster_data = cluster_data[fix_index]

# def fit
##########
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

p0_sig_clust = [max(label_data[:,-1]), np.median(cluster_data),1,min(label_data[:,-1])]
popt_sig_clust, _ = curve_fit(sigmoid, cluster_data, label_data[:,-1], p0=p0_sig_clust)

np.save("popt_sig_clust", popt_sig_clust)

label_data[:, 0] = label_data[:,-1] - sigmoid(cluster_data, *popt_sig_clust)

print("Number of final samples: ", len(image_data))

# convert data to batch form
image_data = image_data.reshape(image_data.shape[0], 1, lx, ly)
label_data = label_data.reshape(label_data.shape[0], ld_arr_size)

# convert data to torch tensors
image_data = torch.from_numpy(image_data.astype(np.float32))
label_data = torch.from_numpy(label_data.astype(np.float32))

torch.save(image_data, "../training_data/image_data_cluster")
torch.save(label_data, "../training_data/label_data_cluster")

gridstates.close()
index_file.close()