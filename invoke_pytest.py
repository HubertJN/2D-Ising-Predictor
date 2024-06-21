# This is a hack to attach the debugger to pytest

import pytest
import sys

# Uncomment this to add a config file to the pytest debug run
#sys.argv.append('config-error.log')

retcode = pytest.main(args=[])


