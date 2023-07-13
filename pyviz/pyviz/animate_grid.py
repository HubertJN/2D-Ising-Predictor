import helper

sim_set = helper.SimulationSet('/home/pgrylls/scratch/code/GPU_Arch_Test_2/configurations/test_input.dat')

print(sim_set.simulations[0].config)

sim_set.simulations[0].load_all_grids()
print(sim_set.simulations[0].all_grids.shape)
print(sim_set.simulations[0].all_grids[0])