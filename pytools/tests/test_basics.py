# Tests for the main menu functionality
from .test_helpers import yeilding_event_loop
from pytools.pyconf.main import ConfigOptions


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