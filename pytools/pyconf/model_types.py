class Simulation():

    def __init__(self):
        self.model_config = {
            'model_id': None,
            'num_concurrent': None,
            'nucleation_threshold': None,
            'grid_size': None,
            'num_iterations': None,
            'iterations': None,
            'iter_per_step': None,
            'seed': None,
            'inv_temp': None,
            'field': None,
            'starting_config': None,
        }

        pass
    
    def array_element_size(self):
        pass

    def generate_file_names(self):
        pass


class Type1(Simulation):

    def __init__(self):
        super().__init__()
        # hard coded information about the model.
        array_element_size = 4 #bytes
        self.model_config['model_id'] = 1
        
        

    def array_size(self):
        pass

class Type2(Simulation):

    def __init__(self):
        super().__init__()
        # hard coded information about the model.
        array_element_size = 4 #bytes
        self.model_config['model_id'] = 2 

    def array_size(self):
        pass


# Dict of all model types that can be used in the simulation
# Models not defined here will not be available for use
ModelTypes = {
    Type1.__name__ : Type1,
    Type2.__name__ : Type2,
    }