import struct
import numpy as np

def write_header(file, info):

    sz_sz = info["sz_sz"]
    endian = info["endian"]
    mag_sz = info["mag_sz"]
    int_sz = info["int_sz"]


    file.write(int_sz.to_bytes(sz_sz, endian))

    file.write(mag_sz.to_bytes(sz_sz, endian))

    fmt = info["mag_fmt_string"]
    check_value = 3.0/32.0
    check_bytes = struct.pack(fmt, check_value)
    file.write(check_bytes)

    vers_bytes = bytes(info["code_version"], 'utf-8') + b'\x00' # Add null term
    print(vers_bytes)


    file.write(vers_bytes)

def write_grid_meta(file, info, grid_meta):

    sz_sz = info["sz_sz"]
    endian = info["endian"]

    n_dims = grid_meta["n_dims"]
    dims = grid_meta["dims"]
    n_conc = grid_meta["n_conc"]

    next_loc = file.tell() + sz_sz*(n_dims + 3)

    file.write(next_loc.to_bytes(sz_sz, endian))

    file.write(n_dims.to_bytes(sz_sz, endian))

    for i in range(n_dims):
        file.write(dims[i].to_bytes(sz_sz, endian))

    file.write(n_conc.to_bytes(sz_sz, endian))

def write_mag_data(file, info, grid_meta, data):
    # Currently write some stuff here - update main writer/reader and duplicate
    # For now assume data contains mag_value and nuc_value which are
    # lists of length n_conc

    sz_sz = info["sz_sz"]
    endian = info["endian"]
    mag_sz = info["mag_sz"]
    fmt = info["mag_fmt_string"]

    print(fmt)
    n_conc = grid_meta["n_conc"]

    next_loc = file.tell() + (sz_sz*2 + mag_sz)*n_conc + sz_sz
    file.write(next_loc.to_bytes(sz_sz, endian))

    for i in range(n_conc):
        file.write(i.to_bytes(sz_sz, endian))
        mag_bytes = struct.pack(fmt, data["mag_value"][i])
        file.write(mag_bytes)
        file.write(data["nuc_value"][i].to_bytes(sz_sz, endian))


def write_grids(file, info, grid_meta, grids):
    # Assume grids is a list of length n_conc of NUMPY arrays of type int!!

    sz_sz = info["sz_sz"]
    mag_sz = info["mag_sz"]
    endian = info["endian"]

    dims = grid_meta["dims"]
    total_sz = grid_meta["total_sz"]
    n_conc = grid_meta["n_conc"]

    dt = info["grid_dtype"]

    next_loc = file.tell()
    # Now the actual grids
    for i in range(n_conc):
        data = np.random.randint(0,2, dims).astype(dt)
        next_loc += (mag_sz * total_sz) + sz_sz
        file.write(next_loc.to_bytes(sz_sz, endian))

        file.write(data.tobytes())
        mag_tmp = np.sum(data)
        print("Mag as written for grid {} is {}".format(i, mag_tmp))


def create_info():
    # Probably should do this better, but for now, create the lengths to match test system

    mag_sz = 4 # Size of floating pt magnetization value
    int_sz = 4 # Size of int, i.e. grid data

    endian = 'little'
    # Format for float packing
    fmt = '<'
    if endian == "big": fmt = '>'

    if mag_sz == 4:
        fmt += 'f'
    elif mag_sz == 8:
        fmt += 'd'

    info = {}
    info["sz_sz"] = 8 # Size of size_t in C on this system ideally
    info["endian"] = endian
    info["mag_sz"] = mag_sz
    info["int_sz"] = int_sz

    info["mag_fmt_string"] = fmt

    grid_fmt_string = "i{}".format(int_sz)
    info["grid_fmt_string"] = grid_fmt_string
    dt = np.dtype(grid_fmt_string)
    info["grid_dtype"] = dt

    _file_leader = "VERSION="
    _leader_len = len(_file_leader)
    with open("VERSION", 'r') as infile:
        vers = infile.read()

    vers = "p-"+vers[_leader_len+1:]
    info["code_version"] = vers[:10]

    return info

def create_dummy_grid_meta():
    
    grid_meta = {}
    grid_meta["n_dims"] = 2

    dims = (100, 100)
    grid_meta["dims"] = dims

    total_sz = dims[0]
    for i in range(1, len(dims)): total_sz *= dims[i]
    grid_meta["total_sz"] = total_sz

    grid_meta["n_conc"] = 5

    return grid_meta


def extract_mag_data(info, grid_meta, grid_data):

    n_conc = grid_meta["n_conc"]

    mag_data = {}
    mag_data["mag_value"] = []
    mag_data["nuc_value"] = []
    for i in range(n_conc):
        mag_tmp = np.sum(grid_data[i])
        nuc_tmp = 0
        if(mag_tmp > (0.5 * grid_meta["total_sz"])): nuc_tmp = 1
        mag_data["mag_value"].append(mag_tmp)
        mag_data["nuc_value"].append(nuc_tmp)

    return mag_data


def create_dummy_grids(info, grid_meta):


    dims = grid_meta["dims"]
    n_conc = grid_meta["n_conc"]

    total_sz = dims[0]
    for i in range(1, len(dims)): total_sz *= dims[i]

    dt = info["grid_dtype"]
    # Dummy random grids of 1 and 0
    grid_data = []
    for i in range(n_conc):
        data = np.random.randint(0,2, dims).astype(dt)
        grid_data.append(data)

    return grid_data

def write_file(filename, info, grid_meta, mag_data, grid_data):

    file = open(filename, "wb")

    write_header(file, info)

    write_grid_meta(file, info, grid_meta)
    write_mag_data(file, info, grid_meta, mag_data)
    write_grids(file, info, grid_meta, grid_data)
 

def example():
    filename = "./testfile.dat"
 
    info = create_info()

    # Replace the following two with actual grid size and grid datas
    grid_meta = create_dummy_grid_meta()
    grid_data = create_dummy_grids(info, grid_meta)

    mag_data = extract_mag_data(info, grid_meta, grid_data)
 
    write_file(filename, info, grid_meta, mag_data, grid_data)

if __name__ =="__main__":

    example()


