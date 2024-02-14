import numpy as np
import struct
from pathlib import Path
from os.path import basename

def read_next_location_item(file, sz_sz, endian):
    """Read single value and return it"""
    next_loc = int.from_bytes(file.read(sz_sz), endian)
    return next_loc

def read_headers(file, sz_sz=8):
    """Read file header info etc, returning dict of sizes etc"""

    endian = 'little' # ToDo use check to verify this...

    nxt = file.read(sz_sz) # Size of grid data
    grid_sz = int.from_bytes(nxt, endian)

    # Format for float unpacking
    mag_sz = 4
    fmt = 'f'
    if mag_sz == 8:
        fmt = 'd'

    check_raw = file.read(mag_sz)
    check = struct.unpack(fmt, check_raw)[0]

    if check != 3.0/32.0: 
        print("ERROR: Magic number does not match - check endianness, float size and data file")
        raise IOError("Wrong magic number")

    version = file.read(11)
    # Version is a byte-string b'...' and final char should be 0. Subscripting -> integer
    if version[-1] != 0:
        print("ERROR: Version string truncated or corrupt")
        raise IOError("Bad version string")
    version = version[0:-1].decode('UTF-8')

    hdr = {}
    hdr["endian"] = endian
    hdr["sz_sz"] = sz_sz
    hdr["mag_sz"] = mag_sz
    hdr["mag_fmt"] = fmt
    hdr["grid_sz"] = grid_sz
    hdr["code_version"] = version

    grid_fmt_string = "i{}".format(grid_sz)
    hdr["grid_fmt_string"] = grid_fmt_string

    # Scan file for locations and store _position to jump to_
    # I.e. include skipping over the next_loc data itself
    next_loc = file.tell()
    hdr["metadata_loc"] = next_loc + sz_sz

    grid_locs = []

    file.seek(0, 2); # Seeks to end
    file_size = file.tell();
    file.seek(next_loc)
    try:
      while next_loc != file_size:
        file.seek(next_loc)
        next_loc = read_next_location_item(file, sz_sz, endian)
        grid_locs.append(next_loc + sz_sz)
    except Exception as e:
      print(e)
 
    hdr["grid_locs"] = grid_locs[:-1] # Last entry is EOF+sz_sz...

    return hdr

def read_grid_meta(file, hdr):
    """Read grid meta data, from file using hdr data"""
    sz_sz = hdr["sz_sz"]
    endian = hdr["endian"]
 
    file.seek(hdr["metadata_loc"])

    n_dims = int.from_bytes(file.read(sz_sz), endian)

    dims = []
    total_sz = 1
    for i in range(n_dims):
        dim = int.from_bytes(file.read(sz_sz), endian)
        dims.append(dim)
        total_sz *= dim

    n_conc = int.from_bytes(file.read(sz_sz), endian)

    meta = {}
    meta["n_dims"] = n_dims
    meta["dims"] = dims
    meta["total_sz"] = total_sz
    meta["n_conc"] = n_conc

    return meta

def read_mag_data(file, hdr, meta):
    """Read magnetization and nucleation grid data, from current file position uding hdr data"""

    file.seek(hdr["mag_data_loc"])

    mag_list = []
    nuc_list = []
   
    endian = hdr["endian"]
    sz_sz = hdr["sz_sz"]
    mag_sz = hdr["mag_sz"]
    mag_fmt = hdr["mag_fmt"]
    n_conc = meta["n_conc"]

    for i in range(n_conc):
        indx = int.from_bytes(file.read(sz_sz), endian)

        mag_raw = file.read(mag_sz)
        mag = struct.unpack(mag_fmt, mag_raw)[0]
        nuc = int.from_bytes(file.read(sz_sz), endian)
        mag_list.append(mag)
        nuc_list.append(nuc)

    mag_data = {}
    mag_data["magnetisations"] = mag_list
    mag_data["nucleations"] = nuc_list

    return mag_data

def read_grids(file, hdr, meta):
    """Read all grids into a list, from current file position using hdr data and meta data"""

    file.seek(hdr["grid_locs"][0])

    grids = []
    for i in range(len(hdr["grid_locs"])):
        grids.append(read_next_grid(file, hdr, meta))
        file.seek(hdr["sz_sz"], 1) # Seek past the next_loc data

    return grids

def read_next_grid(file, hdr, meta):
    """Read a single grid of given size, from current file position, using hdr data and meta data"""

    total_sz = meta["total_sz"]
    grid_sz = hdr["grid_sz"]
    dt = np.dtype(hdr["grid_fmt_string"])
    dims = meta["dims"]
    raw_data = file.read(total_sz * grid_sz)
    data = np.frombuffer(raw_data, dt, count=total_sz)
    data = np.reshape(data, dims)
    print("Calculated magnetization: {}:".format(np.sum(data))) # Print mag again to verify read

    return data

def read_grid_num(file, hdr, meta, num):
    """ Read grid by number from file"""
    # NB uses hdr info on block locations so can call this on any file state

    if num < 0 or num >= meta["n_conc"]:
        raise IndexError("Index not in range")

    file.seek(hdr["grid_locs"][num])
    return read_next_grid(file, hdr, meta)


def read_file_prep(filename, data):

    file = open(filename, "rb")

    sz_sz = 8
    try:
        hdr = read_headers(file, sz_sz)
    except:
        sz_sz = 4
        hdr = read_headers(file, sz_sz)
        # Try again with size 4. If neither 4 or 8, bomb because we're out of ideas

    metadata = read_grid_meta(file, hdr)

    #Sanity checks
    try:
        assert(metadata["n_dims"] <= 3)
        assert(metadata["n_conc"] <= 1000)
    except:
        raise IOError("File data is misread or corrupt")

    data['hdr'] = hdr
    data['metadata'] = metadata

    return file

def read_model_file(filename):

    data = {}
    file = open(filename, "r")
    for line in file.readlines():
        try:
            tmp = line.strip('\n').split('\t')
            tmp2 = tmp[0].split()
            num = int(tmp2[2])
            m_id = tmp2[5]
            # Field order should match C writer code...
            item = {"copy":num, "temp":float(tmp[1]), "field":float(tmp[2]), "start_config":int(tmp[3])}
            data[m_id] = item
        except:
            print("Could not parse model file line {}".format(line))

    return data

def read_file(filename, model_data=None):

    data = {}
    file = read_file_prep(filename, data)

    hdr = data["hdr"]
    metadata = data["metadata"]
    #data["magnetisation"] = read_mag_data(file, hdr, metadata)

    data['grids'] = read_grids(file, hdr, metadata)

    file.close()

    if(model_data):
        model_id = ""
        try:
            #Remove grid_ part, grab 5 digit code
            model_id = basename(filename)[5:10]
            model_meta = model_data[model_id]
            data["model"] = model_meta
        except:
            print("Model data not found for {}".format(model_id))

    return data

if __name__=="__main__":

    # use the output file
    # outnum = 0
    # output_folder = Path(__file__).parent.parent.parent / "grid_binaries" / "output"
    # assert output_folder.exists(), f"{output_folder} folder does not exist"
    # output_file = output_folder / f"grid_0_10000_{outnum}.dat"
    # assert output_file.exists(), f"{output_file} does not exist"

    # use the test file
    input_folder = Path(__file__).parent.parent.parent / "grid_binaries" / "input"
    assert input_folder.exists(), f"{input_folder} folder does not exist"
    input_file = input_folder / "testfile.dat"
    assert input_file.exists(), f"{input_file} does not exist"

    data = read_file(input_file)

    for i in range(len(data["grids"])):

      print("grid num:{}".format(i))
      print(data["grids"][i])

    print("_______________________________________")
    print("Reading grid number 1 manually")

    # Demo non-stream reading, once we've read the header and meta data
    # I.e call read_file_prep and then can read grid by grid, or grab magnetisation etc data
    d_data = {}
    file = read_file_prep(input_file, d_data)
    grid1 = read_grid_num(file, d_data["hdr"], d_data["metadata"], 1)
    print(grid1)

    # Demo reading model data and then a named file
    # Use only if said file exists...
    m_filename = "./grid_binaries/output/grid.meta"
    model_data = read_model_file(m_filename)
    filename = "grid_binaries/output/grid_WUXZE_0_10000.dat"
    data2 = read_file(filename, model_data)
    # data2 now contains file contents and model information
