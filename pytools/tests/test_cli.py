import pytest
from pytools.pyconf.main import event_loop, ConfigOptions
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
    monkeypatch.setattr('builtins.input', lambda _: 'AddModel')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'model-name-1')
    #monkeypatch.setattr('builtins.input', lambda _: 'Type1')
    #yeilding_event_loop(ConfigObj)
    
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
    assert ConfigObj.gpu == None
    #assert ConfigObj.models == {'model-name-1': ModelTypes['Type1']()}
    