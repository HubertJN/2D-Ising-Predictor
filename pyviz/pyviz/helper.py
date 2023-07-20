import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_gif import GIF, capture

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
        self.all_mag_nuc = np.array([i[1] for i in grids_and_mags]).astype(float)
            
    def load_grid_set(self):
        pass

    def load_grid_single(self, file_name):
        with open(file_name, 'r') as f:
            lines = f.readlines()
            get_dims = lines.pop(0)
            dims = get_dims.split()[2:]
            dims = [int(i) for i in dims]
            grids = []
            mag_nuc = []
            for i in range(self.config['num_concurrent']):
                # Copy 1, Mag -0.989800, Nucleated 0
                copyline = lines.pop(0).split()
                mag = float(copyline[3].strip(','))
                nuc = int(copyline[5].strip(','))
                assert (copyline[0], copyline[1].strip(',')) == ("Copy", f"{i+1}"), "File not formatted correctly"
                grids.append(
                    [lines.pop(0).split()
                    for _ in range(dims[0])]
                )
                mag_nuc.append((mag, nuc))
                lines.pop(0)
        return grids, mag_nuc

    def make_figure(self):
        self.maxcols = 4
        cols = min(self.config['num_concurrent'], self.maxcols)
        rows = int(np.ceil(self.config['num_concurrent']/cols))

        self.figure = make_subplots(rows=rows, cols=cols, 
                                    subplot_titles=[f"Copy {i+1}" for i in range(self.config['num_concurrent'])], 
                                    horizontal_spacing=0.1, vertical_spacing=0.1)

        for i in range(self.config['num_concurrent']):
            self.figure.add_trace(
                go.Image(z=self.image_grids[i,0,:,:]),
                row=(i//self.maxcols)+1, col=(i%self.maxcols)+1
            )
    
    def create_layout(self):
        # The buttons dont work either
        # buttons = [
        #     dict(
        #         label='Play',
        #         method='animate',
        #         args=[
        #             None, 
        #             dict(
        #                 frame=dict(duration=50, redraw=False), 
        #                 transition=dict(duration=0),
        #                 fromcurrent=True,
        #                 mode='immediate'
        #             )
        #         ]
        #     ),
        #     dict(
        #         label='Pause',
        #         method='animate',
        #         args=[
        #             [None],
        #             dict(
        #                 frame=dict(duration=0, redraw=False), 
        #                 transition=dict(duration=0),
        #                 fromcurrent=True,
        #                 mode='immediate'
        #             )
        #         ]
        #     )
        # ]


        # Adding a slider
        # This does not work as expected
        # sliders = [{
        #     'yanchor': 'top',
        #     'xanchor': 'left', 
        #     'active': 1,
        #     'currentvalue': {'font': {'size': 16}, 'prefix': 'Steps: ', 'visible': True, 'xanchor': 'right'},
        #     'transition': {'duration': 200, 'easing': 'linear'},
        #     'pad': {'b': 10, 't': 50}, 
        #     'len': 0.9, 'x': 0.15, 'y': 0, 
        #     'steps': [{'args': [[k], {'frame': {'duration': 200, 'easing': 'linear', 'redraw': False},
        #                               'transition': {'duration': 0, 'easing': 'linear'}}], 
        #                               'label': k, 'method': 'animate'} for k in range(len(self.figure.frames) - 1)       
        #         ]
        # }]


        self.figure.update_layout(
            # updatemenus=[
            #     dict(
            #         type='buttons',
            #         showactive=False,
            #         y=0,
            #         x=1.05,
            #         xanchor='left',
            #         yanchor='bottom',
            #         buttons=buttons )
            # ],
            width=800, height=500
        )
    

    def animate_all_grids(self):
        self.figure.write_html(self.path_base/"animation_test.html")

    def animate_grid_set(self):
        pass

    def create_frames(self):
        """Creates a list of frames"""
        frame_list = [
            go.Frame(
                data=[go.Image(z=self.image_grids[i,j,:,:]) for j in range(self.config['num_concurrent'])], 
                traces=[j for j in range(self.config['num_concurrent'])],
            )
            for i in range(self.image_grids.shape[0])
        ]
        self.figure.frames = frame_list
