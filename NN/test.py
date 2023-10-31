import numpy as np

array = np.load("./cnn_loop/minimum_loss.npy")

array = array[array > 0]

print(array)

print(np.min(array), np.argmin(array))