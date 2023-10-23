from functools import partial
import helper
import numpy as np

class ConfigOptions():
    def __init__(self):
        # options is a dict that 
        self.options = {}
        self.config = ""
        self.gpu = None
        self.go_up = False
        self.previous_options = []
        self.ortho_params = {}
        self.sim_class = helper.SimulationSet()
        self.model_weights = {}
        self.gpu_use = {}
        self.optimisable_param = ('num_concurrent', 'grid_size')
        pass
    
    def CreateInitalOptions(self):
        self.options['Exit'] = 'Exit'
        self.options['QueryGPU'] = self.QueryGPU
        self.options['ViewConfig'] = self.ViewConfig
        self.options['CreateConfig'] = self.CreateConfig

    def QueryGPU(self):
        self.gpu = helper.get_cuda_device_specs()
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
    
    def FillGPU(self):
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
            'AutoFill': self.AutoFillGPU,
            'ManualFill': self.ManualFillGPU,
        }
        print("Current GPU use:")
        print(f"GPU Memory: {(self.gpu_use['memory']/self.gpu['free_mem_mb'])*100}%, {self.gpu_use['memory']}MB of {self.gpu['free_mem_mb']}MB")
        print(f"GPU Cores: {(self.gpu_use['cores']/self.gpu['multiprocessors'])*100}%, {self.gpu_use['cores']} of {self.gpu['multiprocessors']}")
        # Print additioanl information here as needed
        pass

    def AutoFillGPU(self):
        print("Autofill not implemented yet.")
        pass

    def ManualFillGPU(self):
        # Set the number of replications for each model sets are automatically shared
        while True:
            # TODO: Swap ', ' to '\n' when base python version is >=3.12
            print(f"Options:\n {  [str(i)+': '+opt+', ' for i, opt in enumerate(self.optimisable_param)] }")
            opt_on = input("Pick an option to fill by: \n")

            if opt_on in self.optimisable_param:
                break
            elif opt_on in [str(i) for i in range(len(self.optimisable_param))]:
                opt_on = self.optimisable_param[int(opt_on)]
                break
            else:
                print("Invalid input. Try again.")
                continue

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
                % Memory Use: P1(Z) % / P1(Zrem) %
                % Core Use: P2(Z) % / P2(Zrem) %
        """

        



        for model in self.sim_class.models.keys():
            single_model_req = self.CalculateSingleModelSize(model)
            unused_gpu = helper.get_unused_gpu(self.gpu, self.gpu_use)
            while True:
                if rep_or_per in ['p', 'per', 'percent', '%']:
                    percent_max = helper.maximise_models_per_gpu(single_model_req, unused_gpu)
                    percent_min = helper.get_one_model_percent(single_model_req, self.gpu)
                    percent_in = input(f"Please enter % GPU use for {model}: ")
                    if percent_in > percent_max:
                        print(f"Max GPU usage for {model} is {percent_max}.")
                        continue
                    elif percent_in < percent_min:
                        print(f"Min GPU usage for {model} is {percent_min}")
                        continue
                    else:
                        replications = helper.percent_to_replications(percent_in, single_model_req, unused_gpu)
                        # Mark the use of the GPU
                        helper.mark_gpu_use(replications, single_model_req, self.gpu_use)
                        self.model_weights[model] = replications
                        break
                
                elif rep_or_per in ['r', 'rep', 'replication']:
                    rep_max = helper.maximise_models_per_gpu(single_model_req, unused_gpu)
                    rep_in = input(f"Please enter replications for {model}: ")
                    if rep_in > rep_max:
                        print(f"Max replications for {model} is {rep_max}.")
                        continue
                    elif rep_in < 0:
                        print(f"Min replications for {model} is 1")
                        continue
                    else:
                        # Mark the use of the GPU
                        helper.mark_gpu_use(replications, single_model_req, self.gpu_use)
                        self.model_weights[model] = rep_in
                        break  
                else:
                    print("Invalid input. Try again.")
            
    
        pass

    def CalculateSingleModelSize(self, model_name, set_key=None, totals={}):
        # Calculate the size of a single model
        # if model is set then calculate the size of one set rep
        if set_key != None:
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



    