import os
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

# parameters
gridstates = open("../../bin/gridstates.bin", "rb")
index_file = open("../../bin/committor_index.bin", "rb")
byte_prefix = 4*(3)
ngrids = 8192
lx = 64
ly = 64
ld_arr_size = 6

# count how many samples
i = 0
while(1):
    index = np.fromfile(index_file, dtype=np.int32, count=3, sep="")
    if index.size < 3:
        break
    committor = np.fromfile(index_file, dtype=np.float64, count=1, sep="")[0]
    std_dev = np.fromfile(index_file, dtype=np.float64, count=1, sep="")[0]
    i += 1

num_init_samp = i

print("Number of initial samples: ", num_init_samp)

index_file.seek(0, os.SEEK_SET)

bytes_per_slice = byte_prefix+ngrids*(lx*ly/8)
# initialised to -1.0 for use later
image_coord = np.full((i, lx*ly, 4), -1.0, dtype=np.float32)
image_neigh = np.full((i, 2, 4*lx*ly), -1.0, dtype=np.float32)
label_data = np.zeros((i, ld_arr_size), dtype=np.float32)

j = 0
ibit=0
ibyte=0
one = np.uint32(1)
blookup = [0, 1]

while(1):
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
    ising_grids = np.reshape(np.array(ising_grids),[lx,ly])

    coord_index = 0
    for x in range(lx):
        for y in range(ly):
                image_coord[j, coord_index, 0] = ising_grids[x, y]
                image_coord[j, coord_index, 1] = x
                image_coord[j, coord_index, 2] = y
                image_coord[j, coord_index, 3] = coord_index
                coord_index += 1

    neigh_idx = 0
    inc_dist = 1.001
    lx_inv = 1/lx
    ly_inv = 1/ly

    for i in image_coord[j]:
        x = i[1]
        y = i[2]
        chosen_index = i[3]
        # x_ij and y_ij with periodic boundaries
        x_ij = image_coord[j][:,1]-x
        y_ij = image_coord[j][:,2]-y
        x_ij = x_ij - lx*np.round(x_ij*lx_inv)
        y_ij = y_ij - ly*np.round(y_ij*ly_inv)

        neighbours = image_coord[j][(np.sqrt((x_ij**2 + y_ij**2)) < inc_dist)]
        neighbours = neighbours[(neighbours[:, 1] > x) | (neighbours[:, 1] < x) | (neighbours[:, 2] > y) | (neighbours[:, 2] < y)]

        for neighbour in range(len(neighbours)):
            image_neigh[j, 0, neigh_idx] = chosen_index
            image_neigh[j, 1, neigh_idx] = neighbours[neighbour, 3]
            neigh_idx += 1

    label_data[j, 0] = committor # label
    label_data[j, 1] = 0.54 # inverse temperature $$(tmp)$$
    label_data[j, 2] = 0.07 # field strength $$(tmp)$$
    label_data[j, 3] = 0.0 # perimeter $$(tmp)$$
    label_data[j, 4] = float(index[2]) # cluster size
    label_data[j, 5] = committor # committor

    j += 1

    if j%1000 == 0:
        print(f"Sample {j}/{num_init_samp}")

try:
    os.remove("../training_data/image_coord_gnn")
    os.remove("../training_data/image_neigh_gnn")
    os.remove("../training_data/label_data_gnn")
except:
    pass

# only save spin value in image_coord
image_coord = image_coord[:, :, 0]
image_coord = np.reshape(image_coord, [image_coord.shape[0], image_coord.shape[1], 1]) 

# remove commitors of 0 and 1
fix_index = (label_data[:, -1] > 0) & (label_data[:, -1] <= 0.99)
image_coord = image_coord[fix_index]
image_neigh = image_neigh[fix_index]
label_data = label_data[fix_index]

print("Number of final samples: ", len(image_coord))

# convert data to torch tensors
image_coord = torch.from_numpy(image_coord.astype(np.float32))
image_neigh = torch.from_numpy(image_neigh.astype(np.int32))
label_data = torch.from_numpy(label_data.astype(np.float32))

torch.save(image_coord, "../training_data/image_coord_gnn")
torch.save(image_neigh, "../training_data/image_neigh_gnn")
torch.save(label_data, "../training_data/label_data_gnn")

gridstates.close()
index_file.close()
