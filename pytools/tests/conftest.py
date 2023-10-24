# Fixtures used in CLI testing

import pytest
from pytools.pyconf.main import ConfigOptions

from .test_helpers import yeilding_event_loop

@pytest.fixture
def MainMenuConfigObj():
    ConfigObj = ConfigOptions()
    ConfigObj.CreateInitalOptions()
    return ConfigObj

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

@pytest.fixture
def BlankModel(MainMenuConfigObj, monkeypatch)