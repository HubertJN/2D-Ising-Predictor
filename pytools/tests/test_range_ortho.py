# Tests for the range fill and orthogonal fill functionality
from .test_helpers import yeilding_event_loop

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
    print(ConfigObj.sim_class.models.keys())
    # Check original model is deleted
    assert ConfigObj.sim_class.models.get('fixture-model') == None
    # Check new models are created (non exhaustive)
    assert ConfigObj.sim_class.models.get('fixture-model-rf-nucleation_threshold-0-1-0_1').get('fixture-model-rf-nucleation_threshold-0_0') != None
    assert ConfigObj.sim_class.models.get('fixture-model-rf-nucleation_threshold-0-1-0_1').get('fixture-model-rf-nucleation_threshold-0_9') != None
    assert ConfigObj.sim_class.models.get('fixture-model-rf-nucleation_threshold-0-1-0_1').get('fixture-model-rf-nucleation_threshold-0_0').model_config['nucleation_threshold'] == 0
    assert ConfigObj.sim_class.models.get('fixture-model-rf-nucleation_threshold-0-1-0_1').get('fixture-model-rf-nucleation_threshold-0_9').model_config['nucleation_threshold'] == 0.9
    assert ConfigObj.sim_class.models.get('fixture-model-rf-nucleation_threshold-0-1-0_1').get('fixture-model-rf-nucleation_threshold-0_5') != None
    assert ConfigObj.sim_class.models.get('fixture-model-rf-nucleation_threshold-0-1-0_1').get('fixture-model-rf-nucleation_threshold-0_5').model_config['nucleation_threshold'] == 0.5

def test_orthogonal_fill(InitObjectConfigObj, monkeypatch):
    ConfigObj = InitObjectConfigObj
    monkeypatch.setattr('builtins.input', lambda _: 'CreateConfig')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'UpdateModel')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'fixture-model')
    yeilding_event_loop(ConfigObj)
    monkeypatch.setattr('builtins.input', lambda _: 'OrthogonalFill')
    yeilding_event_loop(ConfigObj)
    input_gen = ['nucleation_threshold', 'field', 'inv_temperature', 'Done']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    yeilding_event_loop(ConfigObj)
    input_gen = ['Done', '0', '1', '0.1', 'y', '-1', '1', '0.2', 'y', '0', '1', '0.1', 'y', 'y']
    monkeypatch.setattr('builtins.input', lambda _: input_gen.pop(0))
    yeilding_event_loop(ConfigObj)
    # Check original model is deleted
    assert ConfigObj.sim_class.models.get('fixture-model') == None
    # Check new models are created (non exhaustive)
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_0') != None
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_0').model_config['nucleation_threshold'] == 0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_0').model_config['field'] == -1.0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_0').model_config['inv_temperature'] == 0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_9-field-n1_0-inv_temperature-0_0') != None
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_9-field-n1_0-inv_temperature-0_0').model_config['nucleation_threshold'] == 0.9
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_9-field-n1_0-inv_temperature-0_0').model_config['field'] == -1.0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_9-field-n1_0-inv_temperature-0_0').model_config['inv_temperature'] == 0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_9') != None
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_9').model_config['nucleation_threshold'] == 0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_9').model_config['field'] == -1.0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_9').model_config['inv_temperature'] == 0.9
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_8-inv_temperature-0_0') != None
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_8-inv_temperature-0_0').model_config['nucleation_threshold'] == 0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_8-inv_temperature-0_0').model_config['field'] == 0.8
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_8-inv_temperature-0_0').model_config['inv_temperature'] == 0   
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-n1_0-inv_temperature-0_0') != None
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_9-field-0_8-inv_temperature-0_9').model_config['nucleation_threshold'] == 0.9
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_9-field-0_8-inv_temperature-0_9').model_config['field'] == 0.8
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_9-field-0_8-inv_temperature-0_9').model_config['inv_temperature'] == 0.9
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_0-inv_temperature-0_0') != None
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_0-inv_temperature-0_0').model_config['nucleation_threshold'] == 0.0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_0-inv_temperature-0_0').model_config['field'] == 0.0
    assert ConfigObj.sim_class.models.get('fixture-model-of-nucleation_threshold-0-1-0_1-field-n1-1-0_2-inv_temperature-0-1-0_1').get('fixture-model-of-nucleation_threshold-0_0-field-0_0-inv_temperature-0_0').model_config['inv_temperature'] == 0.0
