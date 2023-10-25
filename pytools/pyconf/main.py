from functools import partial
import helper
import numpy as np

class ConfigOptions():
    def __init__(self):
        # options is a dict that

        # debug class variables
        self.call_count_a = 0
        self.call_count_b = 0
        self.call_count_c = 0
        # ========

        self.options = {}
        self.config = ""
        self.gpu = None
        self.go_up = False
        self.previous_options = []
        self.ortho_params = {}
        self.sim_class = helper.SimulationSet()
        self.model_weights = {}
        self.gpu_use = {}
        self.gpu_free = {}
        self.optimisable_param = ('num_concurrent', 'grid_size')
        pass
    
    def CreateInitalOptions(self):
        self.options['Exit'] = 'Exit'
        self.options['QueryGPU'] = self.QueryGPU
        self.options['ViewConfig'] = self.ViewConfig
        self.options['CreateConfig'] = self.CreateConfig

    def QueryGPU(self):
        self.gpu = helper.get_cuda_device_specs()
        for gpu in self.gpu:
            gpu['free_mem_b'] = gpu['free_mem_mb']*1024**2
        self.options['ViewGPU'] = self.ViewGPU
        print ('GPU information has been queried.')

    def ViewGPU(self):
        if self.gpu == None:
            print ('No GPU information has been queried.')
        else:
            ix = 0
            for gpu in self.gpu:
                print(f"GPU:{ix}")
                for key, value in gpu.items():
                    print (f"{key}: {value}")
                print('\n')
                ix += 1
    
    def GetCurrentUseGPU(self):
        # Reset and recalculate
        self.gpu_use = {
            'memory': 0,
            'cores': 0,
        }

        for model_key in self.sim_class.models.keys():
            _totals = self.CalculateTotalModelUse(model_key)
            self.gpu_use['memory'] += _totals['memory']
            self.gpu_free['memory'] = self.gpu[0]['free_mem_b'] - self.gpu_use['memory']
            self.gpu_use['cores'] += _totals['cores']
            self.gpu_free['cores'] = self.gpu[0]['cuda_cores'] - self.gpu_use['cores']

        pass

    def FillGPU(self):
        self.GetCurrentUseGPU()
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
            'AutoFill': partial(self.FillOn, self.AutoFillGPU),
            'ManualFill': partial(self.FillOn, self.ManualFillGPU)
        }
        print("Current GPU use:")
        print(f"GPU Memory: {(self.gpu_use['memory']/self.gpu[0]['free_mem_b'])*100}%, {self.gpu_use['memory']}B of {self.gpu[0]['free_mem_b']}B")
        print(f"GPU Cores: {(self.gpu_use['cores']/self.gpu[0]['cuda_cores'])*100}%, {self.gpu_use['cores']} of {self.gpu[0]['cuda_cores']}")
        # Print additioanl information here as needed
        pass
    

    def FillOn(self, passthrough):
        # Set the number of replications for each model sets are automatically shared
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
        }
        for param in self.optimisable_param:
            self.options[param] = partial(passthrough, param)
        print('Please select a parameter to optimise on:')
        pass

    def AutoFillGPU(self, opt_on):
        print("Autofill not implemented yet.")
        pass
    
    def ManualFillGPU(self, opt_on):

        # Precalculate the size of each model then pass to user
        # Menu entries Should look like:
        """
        #: Optimise `Model Name` on `optimisable_param`:
            Single {model/set} size:
                Memory: X MB, (at current grid size)
                Cores: Y,
            Current Use:
                Grid Size: (X,Y),
                Replications: Z,
                % Memory Use: P1 %
                % Core Use: P2 %
            Max Use (Total / Remaining):
                Grid Size: X.Y <= Nmax / X.Y <= Nrem,
                Replications: Zmax / Zrem,
            Current Use:
                % Memory Use:  %
                % Core Use:  %
        """
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
        }
        for model in self.sim_class.models.keys():
            self.MakeOptMenuEntry(model, opt_on)   
        pass

    def MakeOptMenuEntry(self, model_key, opt_on):
        # Calculate the size of a single model
        is_set = isinstance(self.sim_class.models[model_key], dict)
        single_model_req = self.CalculateSingleModelSize(model_key)
        x_dim, y_dim = single_model_req['grid_dims'][0] # used when model is not a set
        sum_xy = sum(map(lambda x: x[0]*x[1], single_model_req['grid_dims'])) # used when model is a set
        gpu_capacity, mem_limit, core_limit = helper.maximise_models_per_gpu(single_model_req, self.gpu[0])
        gpu_free_capacity, free_mem_limit, free_core_limit  = helper.maximise_models_per_gpu(single_model_req, self.gpu_free)
        
        total_limit = f"{'core' if core_limit < mem_limit else 'memory'}"
        free_limit = f"{'core' if free_core_limit < free_mem_limit else 'memory'}"
        all_rep_req = self.CalculateTotalModelUse(model_key)
        per_cores_in_use = all_rep_req['cores']
        per_mem_in_use = all_rep_req['memory']

        # Need to collapse replications and grid size into one number
        reps = single_model_req['replications']
        maximum_grid_size = (self.gpu[0]['free_mem_b']/single_model_req['replications'])*1024**2 // single_model_req['memory_per_grid_element']
        # Two parter
        maximum_grid_remaining = (self.gpu_free['memory']/single_model_req['replications'])*1024**2 // single_model_req['memory_per_grid_element']
        maximum_grid_remaining += x_dim*y_dim * reps


        menu_str = f"\
Optimise {model_key} on {opt_on}: \n\
    Single {'model' if is_set else 'set'} size: \n\
        Memory: {single_model_req['memory']}B, (at current grid size)\n\
        Cores: {single_model_req['cores']}\n\
    Current Use: \n\
        Grid Size{' (total space for the whole set)' if is_set else ''}: {sum_xy if is_set else x_dim}{'' if is_set else ', '}{'' if is_set else y_dim },\n\
        Replications: {single_model_req['replications']},\n\
        % Memory Use: {(single_model_req['memory']/self.gpu[0]['free_mem_b'])*100}%\n\
        % Core Use: {(single_model_req['cores']/self.gpu[0]['cuda_cores'])*100}%\n\
    Max Use (Total Possible / Remaining): \n\
    Note: this is maximising {opt_on} fixing all other parameters. \n\
    For a empty GPU this model is {total_limit} limited. \n\
    For the free GPU resources this model is {free_limit} limited. \n\
        Grid Size: {maximum_grid_size} / {maximum_grid_remaining},\n\
        Replications: {gpu_capacity} / {gpu_free_capacity},\n\
    Current Use: \n\
        % Memory use by model: {per_mem_in_use}\n\
        % Core use by model: {per_cores_in_use}\n\
            "
        
        if is_set:
            set_param_list = model_key.split('-')[2:]
            for param_range in set_param_list:
                param, start, end, step = param_range.split('-')
                if param == opt_on:
                    self.options[menu_str] = lambda _: print(f"Cannot optimise on {opt_on} as it is part of the range.")
                    return
        else:
            self.options[menu_str] = partial(self.UpdateModelOptimisableParam, menu_str, opt_on, is_set, model_key, maximum_grid_size, gpu_capacity)
        pass
    

    def UpdateModelOptimisableParam(self, menu_str, opt_on, is_set, model_key, grid_max, reps_max):
        # Show the current optimisable parameter numbers using the selected menu entry
        print(menu_str)
        # Get the current optimisable parameter
        while True:
            if opt_on == 'grid_size':
                # Get the new grid size
                x_in = input(f"Please enter a value for grid x: ")
                y_in = input(f"Please enter a value for grid y: ")
                try:
                    x_in = int(x_in)
                    y_in = int(y_in)
                except:
                    print(f"Invalid input. Please enter an integer.")
                    continue

                grid_xy = x_in*y_in
                if grid_xy > grid_max:
                    print(f"Grid size ({grid_xy}={x_in}*{y_in}) too large. Max grid size is {grid_max}")
                    continue
                else:
                    new_value = [x_in, y_in]
            elif opt_on == 'num_concurrent':
                # Get the new number of concurrent replications
                reps_in = input(f"Please enter a value for number of concurrent replications: ")
                try:
                    reps_in = int(reps_in)
                except:
                    print(f"Invalid input. Please enter an integer.")
                    continue
                if reps_in > reps_max:
                    print(f"Number of concurrent replications ({reps_in}) too large. Max number of concurrent replications is {reps_max}")
                    continue
                else:
                    new_value = reps_in
        
            confirm = input(f"Please confirm you want to set {opt_on} to {new_value}. (y/n)")
            if confirm in ['y', 'yes', 'Y', 'Yes']:
                self.previous_options.pop()
                self.go_up = True
                break
        
        # Update the model
        if is_set:
            # Update the set
            for set_key in self.sim_class.models[model_key].keys():
                self.sim_class.models[model_key][set_key].model_config[opt_on] = new_value
            pass
        else:
            self.sim_class.models[model_key].model_config[opt_on] = new_value
            pass


    def CalculateSingleModelSize(self, model_name, set_key=None, totals=None):
        # Calculate the size of a single model
        # if model is set then calculate the size of one set rep

        if totals is None:
            totals = {}

        if set_key is not None:
            model = self.sim_class.models[set_key][model_name]
        else:
            model = self.sim_class.models[model_name]
        if isinstance(model, dict):
            # Calculate the size of a set rep
            for _model_name in model.keys():
                self.CalculateSingeModelSize(_model_name, model_name, totals)
                pass
            pass
        elif isinstance(model, helper.Simulation):
            # Calculate the size of a single model
            for key, value in model.get_system_requirements().items():
                if key in totals.keys():
                    totals[key] += value
                else:
                    totals[key] = value
        return totals
        
    def CalculateTotalModelUse(self, model_key):
        # Single use gives us the size of a single model this can be modified
        # to give totals 
        single_totals = self.CalculateSingleModelSize(model_key)

        # Calculate the total use of the model expand this to keep up with optimisiables 
        totals = {}
        totals['memory'] = single_totals['memory'] * single_totals['replications']
        totals['cores'] = single_totals['cores'] * single_totals['replications']

        return totals


    def CreateConfig(self):
        self.previous_options.append(self.options)

        self.options = {
            'GoBack': self.GoBack,
            'AddModel': self.AddModel,
            'UpdateModel': self.UpdateModel,
        }
        if self.gpu != None and self.sim_class.models != {}:
            self.options['FillGPU'] = self.FillGPU
    
    def AddModel(self):
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
        }

        model_name = input('Please enter the a name for the model: ')
        
        for model in self.sim_class.model_types.keys():
            self.options[model] = partial(self.sim_class.add_model, model, model_name)
        self.go_up = True
    
    def UpdateModel(self):
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
        }
        for model in self.sim_class.models.keys():
            self.options[model] = partial(self.UpdateModelConfig, self.sim_class.models[model])
    
    def UpdateModelConfig(self, model):
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
            'AutoFill': partial(self.AutoFill, model),
            'RangeFill': partial(self.RangeFill, model),
            'OrthogonalFill': partial(self.OrthogonalFill, model)
        }
        for param in model.model_config.keys():
            # model_id is not editable as it is used to identify the model type and could change the rest of the list
            if param in ['model_id']:
                continue
            self.options[param] = partial(self.UpdateModelParam, model, param)

        pass

    def AutoFill(self, model):
        for param in model.model_config.keys():
            # Add non-editable parameters here with justifcation
            # model_id is not editable as it is used to identify the model type and could change the rest of the list
            if param in ['model_id']:
                continue
            self.UpdateModelParam(model, param)
        self.ViewModel(model)
        

    def RangeFill(self, model):
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
        }
        # This function will duplicate a given model and range fill a given parameter
        for param in model.model_config.keys():
            # Add non-editable parameters here with justifcation
            # model_id is not editable as it is used to identify the model type and could change the rest of the list
            if param in ['model_id']:
                continue
            # Add parametrs that cannot be range filled here
            if param in ['num_concurrent', 'starting_config']:
                continue
            
            self.options[param] = partial(self.FillRange, model, param)



    def GetRange(self, param):
        while True:
            start = input(f"Please enter a start value for {param}: ")
            if start in [None, '', ' ']:
                print("Invalid input. Try again.")
                continue
            end = input(f"Please enter a end value for {param}: ")
            if end in [None, '', ' ']:
                print("Invalid input. Try again.")
                continue
            step = input(f"Please enter a step value for {param}: ")
            if step in [None, '', ' ']:
                print("Invalid input. Try again.")
                continue
            
            confirm = input(
                f"Please confirm you want to range fill \n \
                {param} from {start} to {end} in steps of {step}. (y/n)"
                )
            if confirm == 'y':
                break
            else:
                confirm = input("Do you want to try again? (y/n)")
                if confirm == 'y':
                    continue
                else:
                    self.GoBack()
                    return
        return (start, end, step)

    def GetDP(self, start, end, step):
        dp = 0
        if '.' in start:
            dp = len(start.split('.')[-1])
        elif '.' in end:
            dp = max(dp, len(end.split('.')[-1]))
        elif '.' in step:
            dp = max(dp, len(step.split('.')[-1]))
        return dp



    def FillRange(self, model, param):
        # This function will duplicate a given model and update a parameter in a input range
        start, end, step = self.GetRange(param) 
        
        dp = self.GetDP(start, end, step)
        set_name = f"{model.__name__}-rf-{param}-{start.replace('.','_').replace('-', 'n')}-{end.replace('.','_').replace('-', 'n')}-{step.replace('.','_').replace('-', 'n')}"

        for value in np.arange(float(start), float(end), float(step)):
            # Convention is model_name-rf-param_name-value
            if np.isclose(value, 0):
                name = f"{model.__name__}-rf-{param}-{f'{value:.{dp}f}'.replace('.','_').replace('-', '')}" 
            else:
                name = f"{model.__name__}-rf-{param}-{f'{value:.{dp}f}'.replace('.','_').replace('-', 'n')}"
            self.sim_class.duplicate_model(model.__name__, set_name, name)
            self.sim_class.models[set_name][name].model_config[param] = round(value, dp)
            # Value to groupby in the config file for postprocessing
            self.sim_class.models[set_name][name].model_config['set_name'] = f"{model.__name__}-rf-{param}-{start}-{end}-{step}"
        
        if input(f"Do you want to remove the original model called {model.__name__}? (y/n)") == 'y':
            self.sim_class.models.pop(model.__name__)

    def OrthogonalFill(self, model):
        # This function will duplicate a given model and orthogonal fill a given parameter set
        self.options = {
            'GoBack': self.GoBack,
            'Done': partial(self.FillOrthogonal, model)
        }
        
        for param in model.model_config.keys():
            # Add non-editable parameters here with justifcation
            # model_id is not editable as it is used to identify the model type and could change the rest of the list
            if param in ['model_id']:
                continue
            # Add parametrs that cannot be range filled here
            if param in ['num_concurrent', 'starting_config']:
                continue
            if param in [self.ortho_params.keys()]:
                continue
            
            self.options[param] = partial(self.ortho_params.update, {param: []})
        pass

    def FillOrthogonal(self, model):
        # This function will duplicate a given model and update a parameters in a input range
        
        ortho_num = len(self.ortho_params)
        if ortho_num == 0:
            print("No parameters selected.")
            return
        if ortho_num == 1:
            self.FillRange(model, self.ortho_params[0])
        else:
            for param in self.ortho_params:
                self.ortho_params[param] += self.GetRange(param)
            
        self.RecursiveFill(model, ortho_num)
        
        if input(f"Do you want to remove the original model called {model.__name__}? (y/n)") == 'y':
            self.sim_class.models.pop(model.__name__) 
        
        self.ortho_params = {}
        pass
    
    
    def RecursiveFill(self, model, n, param_to_set=[], name_str=""):
        if n != 0:
            param = list(self.ortho_params.keys())[len(self.ortho_params)-n]
            start, end, step = self.ortho_params[param]
            dp = self.GetDP(start, end, step)

            for value in np.arange(float(start), float(end), float(step)):
                if np.isclose(value, 0):
                   name_to_pass = f"-{param}-{f'{value:.{dp}f}'.replace('.','_').replace('-', '')}" 
                else:
                    name_to_pass = f"-{param}-{f'{value:.{dp}f}'.replace('.','_').replace('-', 'n')}"
                value_to_pass = [(param, value)]
                self.RecursiveFill(model, n-1, param_to_set+value_to_pass, name_str+name_to_pass)
        else:
            set_name = f"{model.__name__}-of"
            for key, value in self.ortho_params.items():
                set_name += f"-{key}-{value[0].replace('.','_').replace('-', 'n')}-{value[1].replace('.','_').replace('-', 'n')}-{value[2].replace('.','_').replace('-', 'n')}"
            name = f"{model.__name__}-of{name_str}"
            self.sim_class.duplicate_model(model.__name__, set_name, name)
            for param, value in param_to_set:
                start, end, step = self.ortho_params[param]
                dp = self.GetDP(start, end, step)
                self.sim_class.models[set_name][name].model_config[param] = round(value, dp)
            pass


    def UpdateModelParam(self, model, param):
        _input = input(f"Please enter a value for {param}: ")
        if _input in [None, '', ' ']:
            print(f'Skipping {param}')
        else:
            model.model_config[param] = _input
        pass

    def ViewModel(self, model):
        print(f"Model: {model.__class__.__name__}")
        print('Key: Value')
        for key, value in model.model_config.items():
            print(f"{key}: {value}")

    def ViewConfig(self):
        print(self.config)

    def GoBack(self):
        self.ortho_params = {}
        self.options = self.previous_options.pop()

    def Options(self):
        ix = 1
        temp = {}
        for option in self.options.keys():
            temp[str(ix)] = option
            print (f"{ix}: {option}")
            ix += 1
        while True:
            selection = input('Please select an option: ')

            if str(selection) in temp.keys():
                selection = temp[selection]
            
            if selection in self.options.keys():
                print('\n')
                return self.options[selection]

            else:
                print ('Invalid selection.')
                continue
        


def event_loop(ConfigObj):
     while True:
        print ('Please select an option from the menu below.')
        get_input = ConfigObj.Options()
        if get_input == None:
            pass
        elif get_input == 'Exit':
            break
        else:
            if ConfigObj.go_up:
                ConfigObj.go_up = False
                ConfigObj.GoBack()
            get_input()


if __name__ == '__main__':

    ConfigObj = ConfigOptions()
    ConfigObj.CreateInitalOptions()
    print ('Welcome to the GASP configuration tool.')
   
    event_loop(ConfigObj)



    