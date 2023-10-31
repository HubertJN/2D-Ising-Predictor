import scipy.ndimage as ndimage
import torch
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)
committor_data = torch.load("./committor_data")
grid_data = torch.load("./grid_data")

committor_data = committor_data.numpy()
grid_data = grid_data.numpy()

cluster_choice = 200
grid_choice = 0

grid_index = np.where(committor_data[:,-1] == cluster_choice)
grid_info = grid_data[grid_index][grid_choice,0]

clusters, num = ndimage.label(grid_info)

area = ndimage.sum(grid_info, clusters, index=np.arange(clusters.max() + 1))
largest_cluster_index = np.argmax(area)

R = 64
C = 64

def numofneighbour(mat, i, j):
    count = 0
    # UP
    if (i > 0 and mat[i - 1][j] == largest_cluster_index):
        count+= 1
    # LEFT
    if (j > 0 and mat[i][j - 1] == largest_cluster_index):
        count+= 1
    # DOWN
    if (i < R-1 and mat[i + 1][j] == largest_cluster_index):
        count+= 1
    # RIGHT
    if (j < C-1 and mat[i][j + 1] == largest_cluster_index):
        count+= 1
    return count
 
# Returns sum of perimeter of shapes formed with 1s
def findperimeter(mat):
    perimeter = 0
    # Traversing the matrix and finding ones to
    # calculate their contribution.
    for i in range(0, R):
        for j in range(0, C):
            if (mat[i][j] == largest_cluster_index):
                perimeter += (4 - numofneighbour(mat, i, j))
    return perimeter
 
print(findperimeter(clusters), end="\n")
print(largest_cluster_index)