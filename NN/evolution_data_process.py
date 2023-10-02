import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings("ignore")

def numofneighbour(mat, i, j, largest_cluster_index):
    count = 0
    # UP
    if (i > 0 and mat[i - 1][j] == largest_cluster_index):
        count+= 1
    # LEFT
    if (j > 0 and mat[i][j - 1] == largest_cluster_index):
        count+= 1
    # DOWN
    if (i < 64-1 and mat[i + 1][j] == largest_cluster_index):
        count+= 1
    # RIGHT
    if (j < 64-1 and mat[i][j + 1] == largest_cluster_index):
        count+= 1
    return count
 
# Returns sum of perimeter of shapes formed with 1s
def findperimeter(mat, largest_cluster_index):
    perimeter = 0
    # Traversing the matrix and finding ones to
    # calculate their contribution.
    for i in range(0, 64):
        for j in range(0, 64):
            if (mat[i][j] == largest_cluster_index):
                perimeter += (4 - numofneighbour(mat, i, j, largest_cluster_index))
    return perimeter

gridstates = open('/home/eng/phunsc/PhD_Project/2D_Ising_Project/bin/gridstates.bin', 'rb')
index_file = open('/home/eng/phunsc/PhD_Project/2D_Ising_Project/bin/evolution_index.bin', 'rb')
byte_prefix = 4*(3)
ngrids = 8192
L = 64
i = 0
while(1):
    index = np.fromfile(index_file, dtype=np.int32, count=3, sep='')
    if index.size < 3:
        break
    committor = np.fromfile(index_file, dtype=np.float64, count=1, sep='')[0]
    std_dev = np.fromfile(index_file, dtype=np.float64, count=1, sep='')[0]
    i += 1
print(i)

index_file.seek(0, os.SEEK_SET)

bytes_per_slice = byte_prefix+ngrids*(L*L/8)
grid_data = np.zeros((i,64*64), dtype=np.float32)
committor_data = np.zeros((i, 10), dtype=np.float32)
cluster = np.zeros((i, 1), dtype=np.float32)
j = 0
ibit=0
ibyte=0
one = np.uint32(1)
blookup = [0, 1]

while(1):
    index = np.fromfile(index_file, dtype=np.int32, count=3, sep='')
    if index.size < 3:
        break
    index[0] = index[0]/100
    committor = np.fromfile(index_file, dtype=np.float64, count=1, sep='')[0]
    std_dev = np.fromfile(index_file, dtype=np.float64, count=1, sep='')[0]
    gridstates.seek(int(byte_prefix+bytes_per_slice*index[0]+(L*L/8)*(index[1])), os.SEEK_SET)
    bitgrid = np.fromfile(gridstates, dtype=np.byte, count=64*64, sep='')
    isite=0
    ising_grids = list(range(L*L))
    for ibyte in range(int(L*L/8)):
        for ibit in range(8):
            ising_grids[isite] = blookup[(bitgrid[ibyte] >> ibit) & one]
            isite += 1
            if (isite>L*L):
                break
    grid_info = np.reshape(np.array(ising_grids),[64,64])
    clusters, num = ndimage.label(grid_info)
    area = ndimage.sum(grid_info, clusters, index=np.arange(clusters.max() + 1))
    largest_cluster_index = np.argmax(area)
    perimeter = findperimeter(clusters, largest_cluster_index)
    grid_data[j] = ising_grids
    committor_data[j, 0] = committor
    committor_data[j, 1] = std_dev
    committor_data[j, 8] = perimeter
    committor_data[j, 9] = index[2]
    j += 1

try:
    os.remove('/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/evolution_grid_data')
    os.remove('/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/evolution_committor_data')
except:
    pass

#committor_data = torch.load("./committor_data")
#grid_data = torch.load("./grid_data")

#committor_data = committor_data.numpy()
#grid_data = grid_data.numpy()

# get residual data for committor_data
# def fit
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

def sigmoid2d(X, L1, L2, x0, y0, k1, k2, b):
    x, y = X
    z = L1 / (1 + np.exp(-k1*(x-x0))) * L2 / (1 + np.exp(-k2*(y-y0))) + b
    return z

popt_sig_clust = np.load('popt_sig_clust.npy')
popt_sig_perim = np.load('popt_sig_perim.npy')
popt_sig2d = np.load('popt_sig2d.npy')

## add fits and residuals to committor_data
committor_data[:,2] = sigmoid(committor_data[:,-1], *popt_sig_clust)
committor_data[:,3] = committor_data[:,0] - committor_data[:,2]
committor_data[:,4] = sigmoid(committor_data[:,-2], *popt_sig_perim)
committor_data[:,5] = committor_data[:,0] - committor_data[:,4]
committor_data[:,6] = sigmoid2d([committor_data[:,-1], committor_data[:,-2]], *popt_sig2d)
committor_data[:,7] = committor_data[:,0] - committor_data[:,6]

# convert data to batch form
grid_data = grid_data.reshape(grid_data.shape[0], 1, 64, 64)
committor_data = committor_data.reshape(committor_data.shape[0], 10)

## centre clusters
#for i in range(grid_data.shape[0]):
#    ising_grids = grid_data[i,0]
#    grid_x = np.argmax(np.sum(ising_grids, axis=0))
#    grid_y = np.argmax(np.sum(ising_grids, axis=1))
#    ising_grids = np.roll(ising_grids, 32-grid_y, axis=0)
#    ising_grids = np.roll(ising_grids, 32-grid_x, axis=1)
#    grid_data[i,0] = ising_grids

grid_data = torch.from_numpy(grid_data.astype(np.float32))
committor_data = torch.from_numpy(committor_data.astype(np.float32))

## committor_data is formatted as such
# commitor, standard deviation of committor, cluster fit committor, cluster fit committor correcion, perimeter fit committor, perimeter fit correction, cluster-perimeter fit,
# cluster-perimeter correction, largest cluster size, perimeter of largest cluster

torch.save(grid_data, '/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/evolution_grid_data')
torch.save(committor_data, '/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/evolution_committor_data')

gridstates.close()
index_file.close()