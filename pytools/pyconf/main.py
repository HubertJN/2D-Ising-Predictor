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

    def GetSigFig(self, start, end, step):
        dp = 0
        if '.' in start:
            dp = len(start.split('.')[-1])
        elif '.' in end:
            dp = max(dp, len(end.split('.')[-1]))
        elif '.' in step:
            dp = max(dp, len(step.split('.')[-1]))
        sigfigs = max(len(start.split('.')[0]), len(end.split('.')[0]), len(step.split('.')[0]))+dp+1
        return sigfigs



    def FillRange(self, model, param):
        # This function will duplicate a given model and update a parameter in a input range
        start, end, step = self.GetRange(param)
        # while True:
        #     start = input(f"Please enter a start value for {param}: ")
        #     if start in [None, '', ' ']:
        #         print("Invalid input. Try again.")
        #         continue
        #     end = input(f"Please enter a end value for {param}: ")
        #     if end in [None, '', ' ']:
        #         print("Invalid input. Try again.")
        #         continue
        #     step = input(f"Please enter a step value for {param}: ")
        #     if step in [None, '', ' ']:
        #         print("Invalid input. Try again.")
        #         continue
            
        #     else:
        #         confirm = input(
        #             f"Please confirm you want to range fill \n \
        #             {param} from {start} to {end} in steps of {step}. (y/n)"
        #             )
        #         if confirm == 'y':
        #             break
        #         else:
        #             confirm = input("Do you want to try again? (y/n)")
        #             if confirm == 'y':
        #                 continue
        #             else:
        #                 self.GoBack()
        #                 return
        
        # if '.' in start:
        #     dp = len(start.split('.')[-1])
        # elif '.' in end:
        #     dp = max(dp, len(end.split('.')[-1]))
        # elif '.' in step:
        #     dp = max(dp, len(step.split('.')[-1]))
        # sigfigs = max(len(start.split('.')[0]), len(end.split('.')[0]), len(step.split('.')[0]))+dp+1
        
        sigfigs = self.GetSigFig(start, end, step)


        for value in np.arange(float(start), float(end), float(step)):
            # Convention is model_name-rf-param_name-value
            name = f"{model.__name__}-rf-{param}-{str(value).replace('.','_')[:sigfigs]}"
            self.sim_class.duplicate_model(model.__name__, name)
            self.sim_class.models[name].model_config[param] = value
            # Value to groupby in the config file for postprocessing
            self.sim_class.models[name].model_config['set_name'] = f"{model.__name__}-rf-{param}-{start}-{end}-{step}"
        
        if input(f"Do you want to remove the original model called {model.__name__}? (y/n)") == 'y':
            self.sim_class.models.pop(model.__name__)

    def OrthogonalFill(self, model):
        # This function will duplicate a given model and orthogonal fill a given parameter set
        self.previous_options.append(self.options)
        self.options = {
            'GoBack': self.GoBack,
            'Done': self.FillOrthogonal(model)
        }
        
        for param in model.model_config.keys():
            # Add non-editable parameters here with justifcation
            # model_id is not editable as it is used to identify the model type and could change the rest of the list
            if param in ['model_id']:
                continue
            # Add parametrs that cannot be range filled here
            if param in ['num_concurrent', 'starting_config']:
                continue
            if param in [self.ortho_params]:
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
        
        self.ortho_params = {}
        pass
    
    
    def RecursiveFill(self, model, n, param_to_set=[], name_str=""):
        if n != 0:
            param = self.ortho_params.keys()[len(self.ortho_params)-n]
            start, end, step = self.ortho_params[param]
            sigfigs = self.GetSigFig(start, end, step)

            for value in np.arange(float(start), float(end), float(step)):
                name_to_pass = f"-{param}-{str(value).replace('.','_')[:sigfigs]}",
                value_to_pass = [(param, value)]
                self.RecursiveFill(model, n-1, param_to_set+value_to_pass, name_str+name_to_pass)
        else:
            name = f"{model.__name__}-rf-{name_str}"
            self.sim_class.duplicate_model(model.__name__, name)
            for param, value in param_to_set:
                self.sim_class.models[name].model_config[param] = value
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
        self.ortho_params = []
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



    