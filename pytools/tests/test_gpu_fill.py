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

    assert ConfigObj.gpu_free['cores'] == ConfigObj.gpu[0]['cuda_cores'] - 1000 - 4376


def test_gpu_fill_range_single_model(BlankModel, PopulateModel, monkeypatch):
    ConfigObj = BlankModel
    params = {
        'num_concurrent': 10,
        'nucleation_threshold': 0.9,
        'grid_size': [2000, 2000],
        'iterations': 1000,
        'iter_per_step': 10,
        'seed': 1,
        'inv_temp': 1.5,
        'field': 1.5,
        'starting_config': 1,
    }
    PopulateModel(ConfigObj, 'fixture-model', params)
    # Make a range over the nucleation_threshold
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

    monkeypatch.setattr('builtins.input', lambda _: '1')
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)

    monkeypatch.setattr('builtins.input', lambda _: 'QueryGPU')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'FillGPU')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'ManualFill')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'num_concurrent')
    yeilding_event_loop(ConfigObj)
    input_gen = ['2', '300', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.gpu_use['cores'] == 3000

def test_gpu_fill_range_multi_model(BlankModel, AddModel, PopulateModel, monkeypatch):
    ConfigObj = BlankModel
    params = {
        'num_concurrent': 10,
        'nucleation_threshold': 0.9,
        'grid_size': [200, 200],
        'iterations': 1000,
        'iter_per_step': 10,
        'seed': 1,
        'inv_temp': 1.5,
        'field': 1.5,
        'starting_config': 1,
    }
    PopulateModel(ConfigObj, 'fixture-model', params)
    # Make a range over the nucleation_threshold
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

    monkeypatch.setattr('builtins.input', lambda _: '1')
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)


    AddModel(ConfigObj)
    # Note the previous model is renamed during the range fill so this model also gets called fixture-model
    assert 'fixture-model' in ConfigObj.sim_class.models.keys()
    
    params = {
        'num_concurrent': 10,
        'nucleation_threshold': 0.9,
        'grid_size': [20, 20],
        'iterations': 1000,
        'iter_per_step': 10,
        'seed': 1,
        'inv_temp': 1.5,
        'field': 1.5,
        'starting_config': 1,
    }
    PopulateModel(ConfigObj, 'fixture-model', params)
    # Make a range over the nucleation_threshold
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'fixture-model')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'RangeFill')
    yeilding_event_loop(ConfigObj)
    input_gen = ['field', '1.0', '2.0', '0.1', 'y', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)

    monkeypatch.setattr('builtins.input', lambda _: '1')
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)

    # Do the GPU fill (model 1)
    monkeypatch.setattr('builtins.input', lambda _: 'QueryGPU')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'FillGPU')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'ManualFill')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'num_concurrent')
    yeilding_event_loop(ConfigObj)
    input_gen = ['2', '150', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    # Do the GPU fill (model 2)
    input_gen = ['3', '150', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)

    assert ConfigObj.gpu_use['cores'] == 3000

def test_gpu_fill_ortho_single(BlankModel, PopulateModel, monkeypatch):
    ConfigObj = BlankModel
    params = {
        'num_concurrent': 1,
        'nucleation_threshold': 0.9,
        'grid_size': [100, 100],
        'iterations': 1000,
        'iter_per_step': 10,
        'seed': 1,
        'inv_temp': 1.5,
        'field': 1.5,
        'starting_config': 1,
    }
    PopulateModel(ConfigObj, 'fixture-model', params)
    # Make a range over the nucleation_threshold
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'fixture-model')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'OrthogonalFill')
    yeilding_event_loop(ConfigObj)
    input_gen = ['nucleation_threshold', 'field', 'inv_temp', 'Done']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    input_gen = ['Done', '0', '1', '0.1', 'y', '-1', '1', '0.2', 'y', '0', '1', '0.1', 'y', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)

    monkeypatch.setattr('builtins.input', lambda _: '1')
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    

    monkeypatch.setattr('builtins.input', lambda _: 'QueryGPU')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'FillGPU')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'ManualFill')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'num_concurrent')
    yeilding_event_loop(ConfigObj)
    input_gen = ['2', '5', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.gpu_use['cores'] == 5000
    monkeypatch.setattr('builtins.input', lambda _: 'GoBack')
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)

    monkeypatch.setattr('builtins.input', lambda _: 'ManualFill')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'grid_size')
    yeilding_event_loop(ConfigObj)
    input_gen = ['2', '1500', '1500', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    assert ConfigObj.gpu_use['memory'] == 45000000000
    pass

def test_gpu_fill_ortho_multi():
    pass