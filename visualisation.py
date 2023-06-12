import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

file = open('bin/index.bin', 'rb')

histogram = np.zeros([64*64,2])

while True:
    first = np.fromfile(file, dtype=np.int32, count=3, sep='')
    last = np.fromfile(file, dtype=np.float64, count=1)
    
    # check
    if first.size < 3:
        break
    # proccess data
    histogram[first[2],1] += 1

inv_sum = 0
for i in range(64*64):
    histogram[i,0] = i
    if (histogram[i,1] != 0):
        histogram[i,1] = 1/histogram[i,1]
        inv_sum += histogram[i,1]

histogram[:,1] = histogram[:,1]/inv_sum

file.close()

bin_sum, bin_edges, binnumber = binned_statistic(histogram[:,0], histogram[:,1], statistic='sum', bins=50)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2

plt.stairs(bin_sum, bin_edges)
plt.show()