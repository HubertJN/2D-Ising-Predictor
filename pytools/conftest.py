# Sorting out the path for pytest to find the modules
import os
import sys

sys.path.insert(0, "./pytools")
sys.path.insert(0, "./pytools/pyconf")
sys.path.insert(0, "./tests")

os.environ['running_in_pytest'] = 'True'