import helper
import plotly.graph_objects as go

sim_set = helper.SimulationSet('/home/pgrylls/scratch/code/GPU_Arch_Test_2/configurations/test_input.dat')

# Get the first simulation
test_sim = sim_set.simulations[0]
# Load all the grids
test_sim.load_all_grids()
# Convert the grids to image arrays
test_sim.convert_to_image_array()


frames = [
    go.Frame(
        data=go.Image(z=test_sim.image_grids[i,0,:,:]), 
        layout=go.Layout(
            title_text=f"frame {i}",
            width=600, height=600,
        )
    )
    for i in range(test_sim.image_grids.shape[0])
    ]
data = frames[0].data
layout=go.Layout(
    width=600, height=600,
    title="Ising Model",
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(
            label="Play",
            method="animate",
            args=[None]
        )]
    )]
)

fig = go.Figure(data=data, layout=layout, frames=frames)
fig.show()