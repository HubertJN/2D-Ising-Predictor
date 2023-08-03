import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

gridstates = open('/home/eng/phunsc/PhD_Project/2D_Ising_Project/bin/gridstates.bin', 'rb')
index_file = open('/home/eng/phunsc/PhD_Project/2D_Ising_Project/bin/committor_index.bin', 'rb')
byte_prefix = 4*(3)
ngrids = 8192
L = 64
bytes_per_slice = byte_prefix+ngrids*(L*L/8)
grid_data = np.zeros((1000,64*64), dtype=np.float32)
committor_data = np.zeros((1000, 1), dtype=np.float32)
i = 0
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
    grid_data[i] = ising_grids
    committor_data[i] = committor
    i += 1

try:
    os.remove('/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/grid_data')
    os.remove('/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/committor_data')
except:
    pass
# convert data to batch form
grid_data = grid_data.reshape(grid_data.shape[0], 1, 64, 64)
committor_data = committor_data.reshape(committor_data.shape[0], 1)

grid_data = torch.from_numpy(grid_data.astype(np.float32))
committor_data = torch.from_numpy(committor_data.astype(np.float32))

torch.save(grid_data, '/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/grid_data')
torch.save(committor_data, '/home/eng/phunsc/PhD_Project/2D_Ising_Project/NN/committor_data')
gridstates.close()
index_file.close()