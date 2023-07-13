import numpy as np
from pathlib import Path
import plotly.express as px

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

    def animate_all_grids(self):
        pass

    def animate_grid_set(self):
        pass

    def plot_grid_snapshot(self):
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
    
        
        grid_size = self.config['size_x']*self.config['size_y']
        self.file_names = [
            path_base/Path(f"grid_{self.config['sim_num']}_{grid_size}_{i}.txt") 
            for i in range(0, self.config['iterations'], self.config['iter_per_step'])
        ]

    def load_all_grids(self):
        self.all_grids = np.array([self.load_grid_single(i) for i in self.file_names])
            
    def load_grid_set(self):
        pass

    def load_grid_single(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
            get_dims = lines.pop(0)
            dims = get_dims.split()[2:]
            dims = [int(i) for i in dims]
            grids = []
            for i in range(self.config['num_concurrent']):
                assert lines.pop(0).strip() == f"Copy {i+1}".strip(), "File not formatted correctly"
                grids.append(
                    [lines.pop(0).split()
                    for _ in range(dims[0])]
                )
                lines.pop(0)
        return grids

    def animate_all_grids(self):
        px.imshow(self.all_grids, animation_frame=0, binary_string=True)
        pass

    def animate_grid_set(self):
        pass

    def plot_grid_snapshot(self):
        pass