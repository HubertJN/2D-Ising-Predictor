# Fixtures used in CLI testing
import os
from pytest import fixture
from pytools.pyconf.main import ConfigOptions

from .test_helpers import yeilding_event_loop

os.environ['running_in_pytest'] = 'True'

@fixture
def MainMenuConfigObj():
    ConfigObj = ConfigOptions()
    ConfigObj.CreateInitalOptions()
    return ConfigObj

@fixture
def InitObjectConfigObj(MainMenuConfigObj, monkeypatch):
    """
    This fixture creates a model object with a model named 'fixture-model'
    The menu is returned to the main menu
    """
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    # Next event call requies two input rounds before yielding
    input_gen = ['AddModel', 'fixture-model']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'Type1')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'fixture-model')
    yeilding_event_loop(ConfigObj)
    # Next event call requies many input rounds before yielding
    input_gen = ['AutoFill', '5', '0.9', '10', '10', '1000', '50', '10', '12', '1', '1', '1']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj) 
    monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
    yeilding_event_loop(ConfigObj)
    return ConfigObj

@fixture
def BlankModel(MainMenuConfigObj, monkeypatch):
    """
    This fixture creates a model object with a model named 'fixture-model'
    The menu is returned to the main menu
    """
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: '4')
    yeilding_event_loop(ConfigObj)
    input_gen = ['2', 'fixture-model']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'Type1')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: '1')
    yeilding_event_loop(ConfigObj)
    return ConfigObj

@fixture
def AddModel(monkeypatch):

    def _method(ConfigObj):
        """
        This fixture creates a model object with a model named 'fixture-model-(N+1)'
        """
        assert 'CreateConfig' in ConfigObj.options.keys(), "CreateConfig not in options"

        monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
        yeilding_event_loop(ConfigObj)

        model_base_name = 'fixture-model'
        model_name = model_base_name
        ix = 0
        while True:
            if model_name in ConfigObj.sim_class.models.keys():
                ix += 1
                model_name = model_base_name + f'-{ix}'
            else:
                break
        input_gen = ['AddModel', model_name]
        monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0)) 
        yeilding_event_loop(ConfigObj)
        monkeypatch.setattr('builtins.input', lambda _: 'Type1')
        yeilding_event_loop(ConfigObj)
        monkeypatch.setattr('builtins.input', lambda _: '1')
        yeilding_event_loop(ConfigObj)
        pass
    
    return _method

@fixture
def PopulateModel(monkeypatch):
    def _method(ConfigObj, model_name, model_params = {}):
        assert 'CreateConfig' in ConfigObj.options.keys(), "CreateConfig not in options"
        monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
        yeilding_event_loop(ConfigObj)
        monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
        yeilding_event_loop(ConfigObj)
        monkeypatch.setattr('builtins.input', lambda _: model_name)
        yeilding_event_loop(ConfigObj)
        # Next event call requies many input rounds before yielding
        for _key in model_params.keys():
            if _key in ConfigObj.sim_class.models[model_name].model_config.keys():
                if type(model_params[_key]) == list:
                    input_gen = [_key, *model_params[_key]]
                else:
                    input_gen = [_key, model_params[_key]]
                monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
                yeilding_event_loop(ConfigObj)
                assert ConfigObj.sim_class.models[model_name].model_config[_key] == model_params[_key]
            else:
                raise ValueError(f'Key {_key} not in model_config')
        monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
        yeilding_event_loop(ConfigObj)
        yeilding_event_loop(ConfigObj)
        yeilding_event_loop(ConfigObj)
        
        pass

    return _method