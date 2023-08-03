import helper
import plotly.graph_objects as go
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename='./pyviz.log', encoding='utf-8', level=logging.DEBUG)

sim_set = helper.SimulationSet('/home/pgrylls/scratch/code/GPU_Arch_Test_2/configurations/test_input.dat')

# Get the first simulation
test_sim = sim_set.simulations[0]
# Load all the grids
test_sim.load_all_grids()
# Convert the grids to image arrays
test_sim.convert_to_image_array()

test_sim.make_figure()
test_sim.create_frames()
test_sim.create_layout()
test_sim.animate_all_grids()



# frames = [
#     go.Frame(
#         data=go.Image(z=test_sim.image_grids[i,0,:,:]), 
#         layout=go.Layout(
#             title_text=f"Copy 1, frame {i}, mag {test_sim.all_mag_nuc[i][0][0]}, nuc {test_sim.all_mag_nuc[i][0][1]}",
#             width=600, height=600,
#         )
#     )
#     for i in range(test_sim.image_grids.shape[0])
#     ]
# data = frames[0].data
# layout=go.Layout(
#     width=600, height=600,
#     title="Ising Model",
#     updatemenus=[dict(
#         type="buttons",
#         buttons=[
#             dict(
#             label="Play",
#             method="animate",
#             args=[None]
#         )]
#     )]
# )

# layout=go.Layout(
#     title_text=[f"Copy {j}, frame {i}, mag {self.all_mag_nuc[i][j][0]}, nuc {self.all_mag_nuc[i][j][1]}" for j in range(self.config['num_concurrent'])],
#     width=600, height=600,
# )


# fig = go.Figure(data=data, layout=layout, frames=frames)
# fig.show()