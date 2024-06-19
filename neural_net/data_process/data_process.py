"""
============================================================================================
                                 data_process.py

Python file that processes binary data into data usable by PyTorch. The program loads data
then loops over the binary file populating numpy arrays then the data is saved in the
PyTorch tensor format.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

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
image_data = np.zeros((i,1,lx,ly), dtype=np.float32)
feature_coords = np.full((i, lx*ly, 4), -1.0, dtype=np.float32)
edge_data = np.full((i, 2, 4*lx*ly), -1.0, dtype=np.float32)
label_data = np.zeros((i, ld_arr_size), dtype=np.float32)

j = 0
ibit=0
ibyte=0
one = np.uint32(1)
blookup = [0, 1]

# Main loop
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
    image_data[j] = ising_grids

    coord_index = 0
    for x in range(lx):
        for y in range(ly):
                feature_coords[j, coord_index, 0] = ising_grids[x, y]
                feature_coords[j, coord_index, 1] = x
                feature_coords[j, coord_index, 2] = y
                feature_coords[j, coord_index, 3] = coord_index
                coord_index += 1

    neigh_idx = 0
    inc_dist = 1.001 # Distance for spins to count as neighbours
    lx_inv = 1/lx
    ly_inv = 1/ly

    # Finds neighbours by distance
    for i in feature_coords[j]:
        x = i[1]
        y = i[2]
        chosen_index = i[3]
        # x_ij and y_ij with periodic boundaries
        x_ij = feature_coords[j][:,1]-x
        y_ij = feature_coords[j][:,2]-y
        x_ij = x_ij - lx*np.round(x_ij*lx_inv)
        y_ij = y_ij - ly*np.round(y_ij*ly_inv)

        neighbours = feature_coords[j][(np.sqrt((x_ij**2 + y_ij**2)) < inc_dist)]
        neighbours = neighbours[(neighbours[:, 1] > x) | (neighbours[:, 1] < x) | (neighbours[:, 2] > y) | (neighbours[:, 2] < y)]

        for neighbour in range(len(neighbours)):
            edge_data[j, 0, neigh_idx] = chosen_index
            edge_data[j, 1, neigh_idx] = neighbours[neighbour, 3]
            neigh_idx += 1

    label_data[j, 0] = committor # label
    label_data[j, 1] = 0.54 # inverse temperature $$(tmp)$$
    label_data[j, 2] = 0.07 # field strength $$(tmp)$$
    label_data[j, 3] = 0.0 # special cluster size $$(tmp)$$
    label_data[j, 4] = float(index[2]) # geo cluster size
    label_data[j, 5] = committor # committor

    j += 1

    if j%1000 == 0:
        print(f"Sample {j}/{num_init_samp}")


# only save spin value in feature data
feature_data = feature_coords[:, :, 0]
feature_data = np.reshape(feature_data, [feature_data.shape[0], feature_data.shape[1], 1]) 

# remove commitors of 0 and 1
fix_index = (label_data[:, -1] > 0) & (label_data[:, -1] <= 0.99)
image_data = image_data[fix_index]
feature_data = feature_data[fix_index]
edge_data = edge_data[fix_index]
label_data = label_data[fix_index]

print("Number of final samples: ", len(feature_data))

# convert data to torch tensors
image_data = torch.from_numpy(image_data.astype(np.float32))
feature_data = torch.from_numpy(feature_data.astype(np.float32))
edge_data = torch.from_numpy(edge_data.astype(np.int64))
label_data = torch.from_numpy(label_data.astype(np.float32))

torch.save(image_data, "../training_data/image_data")
torch.save(feature_data, "../training_data/feature_data")
torch.save(edge_data, "../training_data/edge_data")
torch.save(label_data, "../training_data/label_data")

gridstates.close()
index_file.close()
