import struct
import numpy as np

endian = 'little' # ToDo use check to verify this...
sz_sz = 8
bytes_read = 0

outnum = 0
filename = "./output/grid_0_10000_{}.dat".format(outnum)
file = open(filename, "rb")


fst = file.read(sz_sz) # Size of magnetization data
bytes_read += sz_sz
mag_sz = int.from_bytes(fst, endian)
print("Size of mag data (float):", mag_sz)

nxt = file.read(sz_sz) # Size of grid data
bytes_read += sz_sz
grid_sz = int.from_bytes(nxt, endian)
print("Size of grid data (int):", grid_sz)

# Format for float unpacking
fmt = 'f'
if mag_sz == 8:
    fmt = 'd'

check_raw = file.read(mag_sz)
bytes_read += mag_sz
check = struct.unpack(fmt, check_raw)[0]
print("Check value", check)

if check != 3.0/32.0: print("ERRROROR")

file_data=[]# Keep track of the jump offsets, for debugging

next_loc = int.from_bytes(file.read(sz_sz), endian)
bytes_read += sz_sz
file_data.append(next_loc)
print('#',next_loc)

n_dims = int.from_bytes(file.read(sz_sz), endian)
bytes_read += sz_sz
print("n_dims", n_dims)

dims = []
for i in range(n_dims):
  dims.append(int.from_bytes(file.read(sz_sz), endian))
  bytes_read += sz_sz

print("dims", dims)

n_conc = int.from_bytes(file.read(sz_sz), endian)
print("n_concurrent", n_conc)
bytes_read += sz_sz

next_loc = int.from_bytes(file.read(sz_sz), endian)
file_data.append(next_loc)
print("#", next_loc)
bytes_read += sz_sz

# Now we get all of the grids meta-data
for i in range(min(n_conc, 100)):  # Min while developing reader - prevent giant loop if n_conc is misread
    indx = int.from_bytes(file.read(sz_sz), endian)

    mag_raw = file.read(mag_sz)
    mag = struct.unpack(fmt, mag_raw)[0]
    nuc = int.from_bytes(file.read(sz_sz), endian)
    print("grid num:{}, magnetisation:{:.6f}, (nucleated?:{})".format(indx, mag, nuc))
    bytes_read += sz_sz + mag_sz + sz_sz


total_sz = dims[0]
for i in range(1, len(dims)): total_sz *= dims[i]
print("Data per grid:", total_sz)

grid_fmt_string = "i{}".format(grid_sz)
dt = np.dtype(grid_fmt_string)

# Now the actual grids
for i in range(min(n_conc, 100)):  # Min while developing reader - prevent giant loop if n_conc is misread
    data = np.zeros(total_sz)
    next_loc = int.from_bytes(file.read(sz_sz), endian)
    file_data.append(next_loc)
    raw_data = file.read(total_sz * grid_sz)
    bytes_read += sz_sz + total_sz * grid_sz
    data = np.frombuffer(raw_data, dt, count=total_sz)
    data = np.reshape(data, dims)
    print("Calculated magnetization: {}:".format(np.sum(data))) # Print mag again to verify read
