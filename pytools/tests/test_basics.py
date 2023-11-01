# Tests for the main menu functionality
from .test_helpers import yeilding_event_loop
from pytools.pyconf.main import ConfigOptions
import pathlib


def test_main_menu(MainMenuConfigObj):
    ConfigObj = MainMenuConfigObj
    assert ConfigObj.options.keys() == {
        'Exit': 'Exit',
        'QueryGPU': ConfigOptions.QueryGPU,
        'ViewConfig': ConfigOptions.ViewConfig,
        'CreateConfig': ConfigOptions.CreateConfig,
        'WriteConfig': ConfigOptions.WriteConfig,
    }.keys()
    assert ConfigObj.previous_options == []
    assert ConfigObj.go_up == False
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
        'WriteConfig': ConfigOptions.WriteConfig,
    }.keys()
    assert ConfigObj.previous_options == []
    assert ConfigObj.go_up == False
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
        'WriteConfig': ConfigOptions.WriteConfig,
    }.keys()
    assert ConfigObj.gpu == None

def test_write_file(BlankModel, AddModel, PopulateModel, monkeypatch):
    ConfigObj = BlankModel
    AddModel(ConfigObj)
    assert 'fixture-model-1' in ConfigObj.sim_class.models.keys()
    params = {
        'num_concurrent': 10,
        'grid_size': [2000, 2000],
    }
    print(ConfigObj.options.keys())
    PopulateModel(ConfigObj, 'fixture-model', params)
    params = {
        'num_concurrent': 5,
        'grid_size': [200, 200],
    }
    PopulateModel(ConfigObj, 'fixture-model-1', params)

    conf_gen = ['WriteConfig', 'test_config', 'y']
    monkeypatch.setattr('builtins.input', lambda _: conf_gen.pop(0))
    yeilding_event_loop(ConfigObj)

    file_path = pathlib.Path('./configurations/test_config.dat')
    assert file_path.exists() == True

    with open(file_path) as f:
        f.read() == "\
## New model ##\
model_id=1\
num_concurrent=10\
nucleation_threshold=None\
size_x=2000\
size_y=2000\
iterations=None\
iter_per_step=None\
seed=None\
inv_temp=None\
field=None\
starting_config=None\
\
## New model ##\
model_id=1\
num_concurrent=5\
nucleation_threshold=None\
size_x=200\
size_y=200\
iterations=None\
iter_per_step=None\
seed=None\
inv_temp=None\
field=None\
starting_config=None\
"
    file_path.unlink()