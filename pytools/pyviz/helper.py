import struct
import numpy as np
from pathlib import Path
import pygame
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_gif import GIF, capture
import logging
logging.basicConfig(filename='pyviz.log', encoding='utf-8', level=logging.DEBUG)


class SimulationSet():

    def __init__(self, input_file):

        self.parse_config(input_file)

    def parse_config(self, input_file):

        self.read_input(input_file)

        self.simulations = [self.parse_sim_type(config) for config in self.parsed_config]
    
    def parse_sim_type(self, config):
        
        if config['model_id'] == 1:
            return Type1(config)

    def read_input(self, input_file):
            
            with open(input_file, 'r') as f:
                lines = f.readlines()
    
            self.parsed_config = []
            sim_dict = None
            sim_num = 0
            for line in lines:
                # If line is a new model then start a new dictionary
                if line.strip() == '## New model ##'.strip():
                    if sim_dict:
                        # Append previous simulation
                        self.parsed_config.append(sim_dict)
                    # Start new simulation
                    sim_dict = {'sim_num': sim_num}
                else:
                    # Parse line
                    self.parse_line(line, sim_dict)
            # Append last simulation
            self.parsed_config.append(sim_dict)
    
    def parse_line(self, line, sim_dict):
        # Split line into key and value
        key, value = line.split('=')
        # Strip white space
        key = key.strip()
        value = value.strip()
        # Add to dictionary
        if key in ['size_x', 'size_y', 'stream_ix', 'model_id', 'iterations', 'iter_per_step', 'starting_config', 'num_concurrent']:
            sim_dict[key] = int(value)
        elif key in ['inv_temperture', 'field']:
            sim_dict[key] = float(value)
        else:
            Warning(f"Key {key} not recognised")
            sim_dict[key] = value
    
     
class Simulation():
    """
    Class for simulation
    """

    def __init__(self, config):
        self.config = config
        self.generate_file_names()

    def generate_file_names(self):
        pass

    def load_all_grids(self):
        pass

    def load_grid_set(self):
        pass

    def load_grid_single(self):
        pass
    
    def make_figure(self):
        pass

    def animate_all_grids(self):
        pass

    def animate_grid_set(self):
        pass

    def plot_frames(self):
        pass


class Type1(Simulation):
    """
    Class for simulation type 1
    """

    def __init__(self, config):
        super().__init__(config)

    def generate_file_names(self):
        """
        Generate file names for all the simulation output files
        """

        # C code that makes the files:
        # snprintf(filename, sizeof(filename), "../output/grid_%d_%d_%d.txt", stream_ix, grid_size, iteration);
        try:
            path_base = Path("./output/").resolve(strict=True)
        except FileNotFoundError as e1:
            try:
                path_base = Path("../output/").resolve(strict=True)
            except FileNotFoundError as e2:
                try:
                    path_base = Path("../../output/").resolve(strict=True)
                except FileNotFoundError as e3:
                    print("Could not find output directory")
                    print(e1, e2, e3)
                    raise FileNotFoundError
    
        self.path_base = path_base
        grid_size = self.config['size_x']*self.config['size_y']
        self.file_names = [
            path_base/Path(f"grid_{self.config['sim_num']}_{grid_size}_{i}.txt") 
            for i in range(0, self.config['iterations'], self.config['iter_per_step'])
        ]

    def convert_to_image_array(self):
        # This expands the array by one dimention to allow for the rgb values
        # We then multiply by 255 to get the values in the range -255 to 255
        # Finally we add 255 and divide by 2 to get the values in the range 0 to 255
        self.image_grids = np.divide(np.add( 
            np.expand_dims(self.all_grids, axis=4) * np.array([255,255,255]), 
                       255), 2).astype(int)

    def load_all_grids(self):
        grids_and_mags = [self.load_grid_single(i) for i in self.file_names]
        self.all_grids = np.array([i[0] for i in grids_and_mags]).astype(int)
        self.all_mag = np.array([i[1] for i in grids_and_mags]).astype(float)
        self.all_nuc = np.array([i[2] for i in grids_and_mags]).astype(int)
            
    def load_grid_set(self):
        pass

    def load_grid_single(self, file_name):
        endian = 'little' # ToDo use check to verify this...
        sz_sz = 8
        bytes_read = 0

        file = open(file_name, "rb")

        fst = file.read(sz_sz) # Size of magnetization data
        bytes_read += sz_sz
        mag_sz = int.from_bytes(fst, endian)
        logging.info("Size of mag data (float):", mag_sz)

        nxt = file.read(sz_sz) # Size of grid data
        bytes_read += sz_sz
        grid_sz = int.from_bytes(nxt, endian)
        logging.info("Size of grid data (int):", grid_sz)

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
        logging.debug('#',next_loc)

        n_dims = int.from_bytes(file.read(sz_sz), endian)
        bytes_read += sz_sz
        logging.debug("n_dims", n_dims)

        dims = []
        for i in range(n_dims):
            dims.append(int.from_bytes(file.read(sz_sz), endian))
            bytes_read += sz_sz

        logging.info("dims", dims)

        n_conc = int.from_bytes(file.read(sz_sz), endian)
        logging.info("n_concurrent", n_conc)
        bytes_read += sz_sz

        next_loc = int.from_bytes(file.read(sz_sz), endian)
        file_data.append(next_loc)
        logging.debug("#", next_loc)
        bytes_read += sz_sz

        # Now we get all of the grids meta-data
        for i in range(min(n_conc, 100)):  # Min while developing reader - prevent giant loop if n_conc is misread
            indx = int.from_bytes(file.read(sz_sz), endian)

            mag_raw = file.read(mag_sz)
            mag = struct.unpack(fmt, mag_raw)[0]
            nuc = int.from_bytes(file.read(sz_sz), endian)
            logging.info("grid num:{}, magnetisation:{:.6f}, (nucleated?:{})".format(indx, mag, nuc))
            bytes_read += sz_sz + mag_sz + sz_sz


        total_sz = dims[0]
        for i in range(1, len(dims)): total_sz *= dims[i]
        logging.info("Data per grid:", total_sz)

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
            logging.debug("Calculated magnetization: {}:".format(np.sum(data)))

        
        return data, mag, nuc

    def make_figure(self):
        self.create_layout()
        pygame.init()
        size = (self.block_size*self.config.size_x, self.block_size*self.config.size_y)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(f"Sweep : {0:d}")
        self.screen.fill(self.BLACK)
        pygame.display.flip()
        self.clock = pygame.time.Clock()
        self.running = True
        self.advance = True
    
    def create_layout(self):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.block_size = 10
        self.color_map = [self.BLACK, self.WHITE]
        self.igrid = 0
        self.isweep = 0
        
    
    def animate_all_grids(self):
        while True:
            irow_max = self.config.size_x
            icol_max = self.config.size_y
            npix = irow_max * icol_max
            for ix in range(npix):
                irow = ix // icol_max
                icol = ix % icol_max
                pygame.draw.rect(self.screen, self.color_map[self.all_grids[self.igrid][self.isweep][irow][icol]], 
                                 (icol*self.block_size, irow*self.block_size, self.block_size, self.block_size))
            self.process_events()
            # Update and limit frame rate
            time_string = "Grid : %d, Sweep : %.d" % (self.igrid, self.isweep)
            pygame.display.set_caption(time_string)
            pygame.display.flip()
            self.clock.tick()
            if not self.running:
                break
            if (self.advance):
                self.isweep += 1
            
    def process_events(self):
            # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    advance = not advance
                if event.key == pygame.K_LEFT:
                    iframe = max(iframe-1,0)
                    advance = False
                if event.key == pygame.K_RIGHT:
                    iframe += 1
                    advance = False
                if event.key == pygame.K_UP:
                    igrid = min(self.igrid+1,self.config.num_concurrent-1)
                if event.key == pygame.K_DOWN:
                    igrid = max(igrid-1,0)
                if event.key == pygame.K_w:
                    # TODO: Write current active grid to file
                    print(f"Grid snapshot written to pyviz/dump/gridinput.bin")

    def animate_grid_set(self):
        pass

    def create_frames(self):
        """Creates a list of frames"""
        frame_list = [
            go.Frame(
                data=[go.Image(z=self.image_grids[i,j,:,:]) for j in range(self.config['num_concurrent'])], 
                layout=go.Layout(annotations=[
                                                {
                                                    'text':f"Copy {j+1}, frame {i} <br> mag {self.all_mag_nuc[i][j][0]:.2f}, nuc {int(self.all_mag_nuc[i][j][1])}",
                                                    'font': {'size': 8},
                                                    'align': 'center'
                                                    }
                                            for j in range(self.config['num_concurrent'])
                                            ] 
                                ),
                traces=[j for j in range(self.config['num_concurrent'])],
            )
            for i in range(self.image_grids.shape[0])
        ]
        self.figure.frames = frame_list
