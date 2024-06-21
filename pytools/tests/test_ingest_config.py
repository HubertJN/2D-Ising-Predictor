# This script takes a config file and returns the config tool in the state before the last option was selected.
# This is useful for debugging the config tool from a crash.
from .test_helpers import yeilding_event_loop, get_inputs
import sys
import pytest

try:
    sys.argv[1]
    skip = False
except IndexError:
    skip = True

@pytest.mark.skipif(skip, reason="No config file passed")
def test_user_input(MainMenuConfigObj, monkeypatch):
    user_inputs = get_inputs()
    print(user_inputs)
    ix = 0
    while ix < len(user_inputs):
        # This copy the list so that the original is not modified
        popping_inputs = [i for i in user_inputs]
        ConfigObj = MainMenuConfigObj
                          
        try:
            while len(popping_inputs) > (ix+1):
                monkeypatch.setattr('builtins.input', lambda _: popping_inputs.pop(0))
                yeilding_event_loop(ConfigObj)
            break
        except Exception as e:
            print(f"Exception raised at input {ix}: {user_inputs[ix]}")
            ix+=1
            continue
    
    print(f"Program in state before crash, next input{'s were' if len(popping_inputs)>1 else 'was'} {user_inputs}")
    pass
