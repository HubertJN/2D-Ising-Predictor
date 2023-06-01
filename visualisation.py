import matplotlib.pyplot as plt
import numpy as np
import struct

file = open('bin/index.bin', 'rb')

histogram = np.zeros(64*64)

one_index = np.fromfile(file, dtype=np.float64, count=4, sep='')

while True:
    one_index = np.fromfile(file, dtype=np.float64, count=4, sep='')
    # check
    if one_index.size < 4:
        break
    # proccess data
    histogram[int(one_index[3])] += 1

file.close()

plt.hist(histogram)