from .test_helpers import yeilding_event_loop

def test_gpu_fill_blank(BlankModel, monkeypatch):
    ConfigObj = BlankModel
    print(ConfigObj.options.keys())
    monkeypatch.setattr('builtins.input', lambda _: '2')
    yeilding_event_loop(ConfigObj)
    print(ConfigObj.options.keys())
    monkeypatch.setattr('builtins.input', lambda _: '4')
    yeilding_event_loop(ConfigObj)
    print(ConfigObj.options.keys())
    monkeypatch.setattr('builtins.input', lambda _: '4')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'ManualFill')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: '2')
    yeilding_event_loop(ConfigObj)
    input_gen = ['2', '10', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: '1')
    yeilding_event_loop(ConfigObj)

    assert ConfigObj.sim_class.models['fixture-model'].model_config['num_concurrent'] == 10


def test_gpu_fill_two_models(BlankModel, AddModel, PopulateModel, monkeypatch):
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

    assert ConfigObj.sim_class.models['fixture-model'].model_config['num_concurrent'] == 10
    assert ConfigObj.sim_class.models['fixture-model-1'].model_config['num_concurrent'] == 5
    assert ConfigObj.sim_class.models['fixture-model'].model_config['grid_size'] == [2000, 2000]
    assert ConfigObj.sim_class.models['fixture-model-1'].model_config['grid_size'] == [200, 200]

    # Get to the GPU fill menu
    monkeypatch.setattr('builtins.input', lambda _: '2')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: '4')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: '4')
    yeilding_event_loop(ConfigObj)
    # Select manual fill, then replications
    monkeypatch.setattr('builtins.input', lambda _: 'ManualFill')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'num_concurrent')
    yeilding_event_loop(ConfigObj)
    get_pre_menu_options = ConfigObj.options.keys()
    # Set replications to 1000
    input_gen = ['2', '1000', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    get_post_menu_options = ConfigObj.options.keys()
    # Check the model had been updated
    assert ConfigObj.sim_class.models['fixture-model'].model_config['num_concurrent'] == 1000

    # Check that the menu options have changed to reflect the new replications
    assert get_pre_menu_options != get_post_menu_options

    input_gen = ['3', '4376', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.sim_class.models['fixture-model-1'].model_config['num_concurrent'] == 4376

    get_post_menu_options_2 = ConfigObj.options.keys()
    # Check that the menu options have changed to reflect the new replications
    assert get_post_menu_options != get_post_menu_options_2

    print(ConfigObj.options.keys())
    assert False

