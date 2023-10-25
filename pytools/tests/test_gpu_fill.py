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

