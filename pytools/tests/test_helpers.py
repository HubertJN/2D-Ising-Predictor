# Libary of helpers for testing the CLI

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
