import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

gridstates = open('/home/eng/phunsc/PhD_Project/2D_Ising_Project/bin/gridstates.bin', 'rb')
index_file = open('/home/eng/phunsc/PhD_Project/2D_Ising_Project/bin/committor_index.bin', 'rb')
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
committor_data = np.zeros((i, 5), dtype=np.float32)
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
    grid_data[j] = ising_grids
    committor_data[j, 0] = committor
    committor_data[j, 3] = std_dev
    committor_data[j, 4] = index[2]
    j += 1

try:
    os.remove('/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/grid_data')
    os.remove('/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/committor_data')
except:
    pass

# get residual data for committor_data
# def fit

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
p0 = [max(committor_data[:,0]), np.median(committor_data[:,-1]),1,min(committor_data[:,0])]

popt, _ = curve_fit(sigmoid, committor_data[:,-1], committor_data[:,0], p0=p0)

# fix data
fix_index = (committor_data[:, 0] >= 0.01) & (committor_data[:, 0] <= 0.99)
grid_data = grid_data[fix_index]
committor_data = committor_data[fix_index]

# converst committor_data to residual
committor_data[:,1] = sigmoid(committor_data[:,-1], *popt)
committor_data[:,2] = committor_data[:,0] - committor_data[:,1]

# convert data to batch form
grid_data = grid_data.reshape(grid_data.shape[0], 1, 64, 64)
committor_data = committor_data.reshape(committor_data.shape[0], 5)

grid_data = torch.from_numpy(grid_data.astype(np.float32))
committor_data = torch.from_numpy(committor_data.astype(np.float32))

torch.save(grid_data, '/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/grid_data')
torch.save(committor_data, '/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/committor_data')

gridstates.close()
index_file.close()

plt.scatter(committor_data[:,2], committor_data[:,0], s=0.5)
plt.show()
