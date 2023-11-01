# Tests for the manual fill functionality

from .test_helpers import yeilding_event_loop
from pytools.pyconf.main import ConfigOptions
from pytools.pyconf.model_types import ModelTypes

def test_add_model(MainMenuConfigObj, monkeypatch):
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    # Next event call requies two input rounds before yielding
    input_gen = ['AddModel', 'model-name-1']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    
    
    # Check Keys
    keys_to_check = {'GoBack': ConfigOptions.GoBack } | ModelTypes
    assert ConfigObj.options.keys() == keys_to_check.keys()
    # Check Previous Options Keys
    assert ConfigObj.previous_options[-1].keys() == {
        'GoBack': ConfigOptions.GoBack,
        'AddModel': ConfigOptions.AddModel,
        'UpdateModel': ConfigOptions.UpdateModel,
    }.keys()
    assert ConfigObj.previous_options[-2].keys() == {
        'Exit': 'Exit',
        'QueryGPU': ConfigOptions.QueryGPU,
        'ViewConfig': ConfigOptions.ViewConfig,
        'CreateConfig': ConfigOptions.CreateConfig,
        'WriteConfig': ConfigOptions.WriteConfig,
    }.keys()
    assert ConfigObj.gpu is None

    monkeypatch.setattr('builtins.input', lambda _: 'Type1')
    yeilding_event_loop(ConfigObj)
    
    assert ConfigObj.sim_class.models.get('model-name-1') != None
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['model_id'] == 1

    # Check Keys
    keys_to_check = {'GoBack': ConfigOptions.GoBack } | ModelTypes
    assert ConfigObj.options.keys() == {
        'GoBack': ConfigOptions.GoBack,
        'AddModel': ConfigOptions.AddModel,
        'UpdateModel': ConfigOptions.UpdateModel,
    }.keys()
    assert ConfigObj.previous_options[-1].keys() == {
        'Exit': 'Exit',
        'QueryGPU': ConfigOptions.QueryGPU,
        'ViewConfig': ConfigOptions.ViewConfig,
        'CreateConfig': ConfigOptions.CreateConfig,
        'WriteConfig': ConfigOptions.WriteConfig,
    }.keys()
    assert ConfigObj.gpu is None


def test_add_two_models(MainMenuConfigObj, monkeypatch):
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    # Next event call requies two input rounds before yielding
    input_gen = ['AddModel', 'model-name-1']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'Type1')
    yeilding_event_loop(ConfigObj)

    # Add a second model
    input_gen = ['AddModel', 'model-name-2']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)

    monkeypatch.setattr('builtins.input', lambda _: 'Type2')
    yeilding_event_loop(ConfigObj)

    assert ConfigObj.sim_class.models.get('model-name-2') != None
    assert ConfigObj.sim_class.models.get('model-name-2').model_config['model_id'] == 2

    monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.options.keys() == {
        'GoBack': ConfigOptions.GoBack,
        'model-name-1': ConfigObj.sim_class.models.get('model-name-1'),
        'model-name-2': ConfigObj.sim_class.models.get('model-name-2'),
        }.keys()
    assert ConfigObj.previous_options[-1].keys() == {
        'GoBack': ConfigOptions.GoBack,
        'AddModel': ConfigOptions.AddModel,
        'UpdateModel': ConfigOptions.UpdateModel,
    }.keys()
    assert ConfigObj.previous_options[-2].keys() == {
        'Exit': 'Exit',
        'QueryGPU': ConfigOptions.QueryGPU,
        'ViewConfig': ConfigOptions.ViewConfig,
        'CreateConfig': ConfigOptions.CreateConfig,
        'WriteConfig': ConfigOptions.WriteConfig,
    }.keys()

def test_autofill(MainMenuConfigObj, monkeypatch):
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    # Next event call requies two input rounds before yielding
    input_gen = ['AddModel', 'model-name-1']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'Type1')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'model-name-1')
    yeilding_event_loop(ConfigObj)
    # Next event call requies many input rounds before yielding
    input_gen = ['AutoFill', '1', '5', '0.9', '10', '10', '1000', '10', '12', '1', '1', '1']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['model_id'] == 1
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['model_itask'] == '1'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['num_concurrent'] == '5'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['nucleation_threshold'] == '0.9'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['grid_size'] == ['10', '10']
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['iterations'] == '1000'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['iter_per_step'] == '10'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['seed'] == '12'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['inv_temp'] == '1'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['field'] == '1'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['starting_config'] == '1'

def test_with_rangefill():
    pass