import struct
import numpy as np
from pathlib import Path
import pygame
import logging

import grid_reader

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
        # snprintf(filename, sizeof(filename), "../output/grid_%d_%d_%d.dat", stream_ix, grid_size, iteration);
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
            path_base/Path(f"grid_{self.config['sim_num']}_{grid_size}_{i}.dat") 
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
        grids_and_mags = [self.load_grid_set(i) for i in self.file_names]
        self.all_grids = np.array([i[0] for i in grids_and_mags]).astype(int)
        self.all_mag = np.array([i[1] for i in grids_and_mags]).astype(float)
        self.all_nuc = np.array([i[2] for i in grids_and_mags]).astype(int)
            
    def load_grid_set(self, file_name):
        # Loads all the grid replications (and their meta data) from an output file
        all_data = grid_reader.read_file(file_name)

        return all_data["grids"], all_data["magnetisation"]["magnetisations"], all_data["magnetisation"]["nucleations"]

    def make_figure(self):
        self.create_layout()
        pygame.display.init()
        pygame.init()
        size = (self.block_size*self.config['size_x'], self.block_size*self.config['size_y'])
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
        self.color_map = [self.BLACK, self.WHITE, self.BLACK] # 0 = black, 1 = white, -1 = black (this gaurds against 0 or -1 being the -ve spin)
        self.igrid = 0
        self.isweep = 0
        
    def animate_all_grids(self):
        number_of_frames = len(self.all_grids)
        self.make_figure()
        while True:
            irow_max = self.config['size_x']
            icol_max = self.config['size_y']
            npix = irow_max * icol_max
            for ix in range(npix):
                irow = ix // icol_max
                icol = ix % icol_max
                pixel = self.all_grids[self.isweep][self.igrid][irow][icol]
                pygame.draw.rect(self.screen, self.color_map[pixel], 
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
                if self.isweep == number_of_frames:
                    self.isweep = 0
            
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
                    self.igrid = min(self.igrid+1, self.config['num_concurrent']-1)
                if event.key == pygame.K_DOWN:
                    self.igrid = max(self.igrid-1,0)
                if event.key == pygame.K_w:
                    # TODO: Write current active grid to file
                    print(f"Grid snapshot written to pyviz/dump/gridinput.bin")

    def animate_grid_set(self):
        pass
