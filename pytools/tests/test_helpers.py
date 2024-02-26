# Libary of helpers for testing the CLI

import sys

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

def get_inputs():
    with open(sys.argv[1], 'r') as log_file:
        lines = log_file.readlines()
    input_lines = list(filter(lambda x: 'User Input: ' in x, lines))
    inputs = list(map(lambda x: x.split('User Input: ')[1].strip('\n'), input_lines))
    return inputs