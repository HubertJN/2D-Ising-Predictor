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
        self.sim_class = helper.SimulationSet()
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
                print(f"GPU:1{ix}")
                for key, value in gpu.items():
                    print (f"{key}: {value}")
                print('\n')
                ix += 1
    
    def CreateConfig(self):
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
            'AddModel': self.AddModel,
            'UpdateModel': self.UpdateModel,
        }
    
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
            'OrthogonalFill': self.OrthogonalFill,
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

        pass

    def FillRange(self, model, param):
        # This function will duplicate a given model and update a parameter in a input range

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
            
            else:
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
        
        # TODO: Check the dp of the three inputs and truncate to the lowest dp to avoid overly long model names

        for value in np.arange(float(start), float(end), float(step)):
            # Convention is model_name-rf-param_name-value
            name = f"{model.__name__}-rf-{param}-{str(value).replace('.','_')}"
            self.sim_class.duplicate_model(model.__name__, name)
            self.sim_class.models[name].model_config[param] = value
        
        if input(f"Do you want to remove the original model called {model.__name__}? (y/n)") == 'y':
            self.sim_class.models.pop(model.__name__)

    def OrthogonalFill(self):
        # This function will duplicate a given model and orthogonal fill a given parameter set
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



    