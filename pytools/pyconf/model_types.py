import numpy as np
class Simulation():

    def __init__(self, model_name):
        self.__name__ = model_name
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

    def generate_file_names(self):
        pass
    
    # functions to return the system requirements of the model
    def get_array_size(self):
        if self.model_config['grid_size'] is None:
            grid_size = None
        else:
            grid_size = self.model_config['grid_size'][0] * self.model_config['grid_size'][1]
        return grid_size

    def get_array_element_size(self):
        assert self.array_element_size is not None, "Array element size not set."
        return self.array_element_size

    def get_num_concurrent(self):
        if self.model_config['num_concurrent'] is None:
            num_concurrent = None
        else:
            num_concurrent = self.model_config['num_concurrent']
        return num_concurrent

    def get_threads_per_concurrent(self):
        # TODO: is this going to raise the correct error?
        assert self.threads_per_concurrent is not None, "Threads per concurrent not set."
        return self.threads_per_concurrent

    def get_system_requirements(self):
        # If a resource is not needed or undefined, set it to None
        sys_req = {}
        # Bytes
        try:
            sys_req['grid_dims'] = [self.model_config['grid_size']]
        except TypeError:
            sys_req['grid_dims'] = [None]
        try:
            sys_req['memory'] = self.get_array_size() * self.get_array_element_size()
        except TypeError:
            sys_req['memory'] = None
        try:
            sys_req['shared_memory'] = [self.get_array_size() * self.get_array_element_size()]
        except TypeError:
            sys_req['shared_memory'] = [None]
        try:
            sys_req['replications'] = self.get_num_concurrent()
        except TypeError:
            sys_req['replications'] = None
        try:
            sys_req['multiprocessors'] = self.get_num_concurrent() * self.get_threads_per_concurrent()
        except TypeError:
            sys_req['multiprocessors'] = None
        try:
            sys_req['threads_per_block'] = self.get_threads_per_concurrent()
        except TypeError:
            sys_req['threads_per_block'] = None
        try:
            sys_req['memory_per_grid_element'] = self.get_array_element_size()
        except TypeError:
            sys_req['memory_per_grid_element'] = None
        
        return sys_req


class Type1(Simulation):

    def __init__(self, model_name):
        super().__init__(model_name)
        # hard coded information about the model.
        self.array_element_size = np.array([1,2,3], dtype=np.intc).itemsize #bytes
        self.threads_per_concurrent = 1
        self.model_config['model_id'] = 1
        
        

    def array_size(self):
        pass

class Type2(Simulation):

    def __init__(self, model_name):
        super().__init__(model_name)
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