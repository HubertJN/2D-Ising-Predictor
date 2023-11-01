# Sorting out the path for pytest to find the modules
import os
import sys
import pytest

sys.path.insert(0, "./pytools")
sys.path.insert(0, "./pytools/pyconf")
sys.path.insert(0, "./tests")

os.environ['running_in_pytest'] = 'True'

# This skipps tests that require cli input
# def pytest_addoption(parser):
#     parser.addoption(
#         "--configfile", 
#         action="store",
#         help="run tests that require a configfiel file passed by cli"
#     )


# def pytest_configure(config):
#     # register an additional marker
#     configfile.addinivalue_line(
#         "markers", "configfile: mark test as requiring a config file"
#     )


# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--configfile"):
#         # --configfile given in cli: do not skip configfile dependent tests
#         return
#     skip_cli = pytest.mark.configfile(reason="need --configfile option to run")
#     for item in items:
#         if "configfile" in item.keywords:
#             item.add_marker(skip_cli)
# ===================================================================================================