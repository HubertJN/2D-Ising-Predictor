import pytest
from pytools.pyconf.main import ConfigOptions
from pytools.pyconf.model_types import ModelTypes


def yeilding_event_loop(ConfigObj):
    get_input = ConfigObj.Options()
    if get_input == None:
        pass
    elif get_input == 'Exit':
        raise SystemExit
    else:
        if ConfigObj.go_up:
            ConfigObj.go_up = False
            ConfigObj.GoBack()
        get_input()


@pytest.fixture
def MainMenuConfigObj():
    ConfigObj = ConfigOptions()
    ConfigObj.CreateInitalOptions()
    return ConfigObj


def test_main_menu(MainMenuConfigObj):
    ConfigObj = MainMenuConfigObj
    assert ConfigObj.options.keys() == {
        'Exit': 'Exit',
        'QueryGPU': ConfigOptions.QueryGPU,
        'ViewConfig': ConfigOptions.ViewConfig,
        'CreateConfig': ConfigOptions.CreateConfig,
    }.keys()
    assert ConfigObj.previous_options == []
    assert ConfigObj.go_up == False
    assert ConfigObj.config == ""
    assert ConfigObj.gpu == None

def test_query_gpu(MainMenuConfigObj, monkeypatch):
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'QueryGPU')
    yeilding_event_loop(ConfigObj)
    assert 'ViewGPU' in ConfigObj.options.keys()
    assert ConfigObj.previous_options == []
    assert ConfigObj.gpu != None


def test_view_config(MainMenuConfigObj, monkeypatch):
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'ViewConfig')
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.options.keys() == {
        'Exit': 'Exit',
        'QueryGPU': ConfigOptions.QueryGPU,
        'ViewConfig': ConfigOptions.ViewConfig,
        'CreateConfig': ConfigOptions.CreateConfig,
    }.keys()
    assert ConfigObj.previous_options == []
    assert ConfigObj.go_up == False
    assert ConfigObj.config == ""
    assert ConfigObj.gpu == None


def test_create_config(MainMenuConfigObj, monkeypatch):
    ConfigObj = MainMenuConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    # Check Keys
    assert ConfigObj.options.keys() == {
        'GoBack': ConfigOptions.GoBack,
        'AddModel': ConfigOptions.AddModel,
        'UpdateModel': ConfigOptions.UpdateModel,
    }.keys()
    # Check Previous Options Keys
    assert ConfigObj.previous_options[-1].keys() == {
        'Exit': 'Exit',
        'QueryGPU': ConfigOptions.QueryGPU,
        'ViewConfig': ConfigOptions.ViewConfig,
        'CreateConfig': ConfigOptions.CreateConfig,
    }.keys()
    assert ConfigObj.config == ""
    assert ConfigObj.gpu == None



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
    }.keys()
    assert ConfigObj.config == ""
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
    }.keys()
    assert ConfigObj.config == ""
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
    input_gen = ['AutoFill', '5', '0.9', '100', '1000', '50', '10', '12', '1', '1', '1']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['model_id'] == 1
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['num_concurrent'] == '5'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['nucleation_threshold'] == '0.9'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['grid_size'] == '100'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['num_iterations'] == '1000'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['iterations'] == '50'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['iter_per_step'] == '10'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['seed'] == '12'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['inv_temp'] == '1'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['field'] == '1'
    assert ConfigObj.sim_class.models.get('model-name-1').model_config['starting_config'] == '1'

# Below this point we need a prebuilt model to test this fixture meets that need
@pytest.fixture
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
    input_gen = ['AutoFill', '5', '0.9', '100', '1000', '50', '10', '12', '1', '1', '1']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj) 
    monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
    yeilding_event_loop(ConfigObj)
    return ConfigObj

def test_range_fill(InitObjectConfigObj, monkeypatch):
    ConfigObj = InitObjectConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'fixture-model')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'RangeFill')
    yeilding_event_loop(ConfigObj)
    input_gen = ['nucleation_threshold', '0', '1', '0.1', 'y', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.sim_class.models.get('fixture-model') == None

    


