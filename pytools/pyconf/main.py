import helper


class ConfigOptions():
    def __init__(self):
        # options is a dict that 
        self.options = {}
        self.config = ""
        self.gpu = None
        pass
    
    def CreateInitalOptions(self):
        self.options['Exit'] = 'Exit'
        self.options['QueryGPU'] = self.QueryGPU
        self.options['ViewConfig'] = self.ViewConfig

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
    
    def ViewConfig(self):
        print(self.config)

    def GoBack(self):
        return None

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
        
            

if __name__ == '__main__':

    ConfigObj = ConfigOptions()
    ConfigObj.CreateInitalOptions()
    print ('Welcome to the GASP configuration tool.')
   
    while True:
        print ('Please select an option from the menu below.')
        get_input = ConfigObj.Options()
        if get_input == None:
            pass
        elif get_input == 'Exit':
            break
        else:
            get_input()



    