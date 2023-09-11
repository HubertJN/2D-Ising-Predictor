import helper
import numpy as np
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename='pyviz.log', encoding='utf-8', level=logging.DEBUG)

logging.info("Starting pyviz ipynb style")

sim_set = helper.SimulationSet('/home/pgrylls/scratch/code/GPU_Arch_Test_2/configurations/test_input.dat')
test_sim = sim_set.simulations[0]

test_sim.load_all_grids()
test_sim.animate_all_grids()