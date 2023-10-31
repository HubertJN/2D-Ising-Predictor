import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.linewidth'] = 0.5

# plot or evolution?
plot = True
evolution = False

# 1) Model

from nn_module import ConvNet
    
device = torch.device('cpu')

net_base = ConvNet().to(device)
net_lcs = ConvNet().to(device)
net_lcs_p = ConvNet().to(device)

PATH = './model_base.pth'
checkpoint_model = torch.load(PATH)
net_base.load_state_dict(checkpoint_model['model_state_dict'])
net_base.eval()
print("Loaded NN")

PATH = './model_lcs.pth'
checkpoint_model = torch.load(PATH)
net_lcs.load_state_dict(checkpoint_model['model_state_dict'])
net_lcs.eval()
print("Loaded NN")

PATH = './model_lcs_p.pth'
checkpoint_model = torch.load(PATH)
net_lcs_p.load_state_dict(checkpoint_model['model_state_dict'])
net_lcs_p.eval()
print("Loaded NN")

# create color map to have only outline for up spins
ncolors = 256
color_array = np.zeros([ncolors, 4])

color_array[:,-1] = np.flip(np.linspace(1.0,0.0, ncolors))
color_array[:,0] = 0
color_array[:,1] = 0
color_array[:,2] = 0

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='bw',colors=color_array)
plt.register_cmap(cmap=map_object)

committor_data = torch.load("./committor_data")
grid_data = torch.load("./grid_data")
evolution_committor_data = torch.load("./evolution_committor_data")
evolution_grid_data = torch.load("./evolution_grid_data")

committor_data = committor_data.numpy()
grid_data = grid_data.numpy()
evolution_committor_data = evolution_committor_data.numpy()
evolution_grid_data = evolution_grid_data.numpy()

# Ising grid
if plot==True:
    cluster_choice = 300
    for grid_choice in range(3):
        grid_index = np.where(committor_data[:,-1] == cluster_choice)
        grid_info = grid_data[grid_index][grid_choice,0]
        img = torch.from_numpy(grid_info.astype(np.float32))
        img = torch.reshape(img, [1,1,64,64])

        img.requires_grad_()
        scores = net_base(img)
        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]
        score_max.backward()
        saliency, _ = torch.max(img.grad.data.abs(),dim=1)
        smooth_saliency = gaussian_filter(saliency[0], sigma=1)

        x,y = np.meshgrid(np.arange(grid_info.shape[1]),np.arange(grid_info.shape[0]))
        plt.pcolor(x,y,smooth_saliency,norm=colors.LogNorm(vmin=smooth_saliency.min(), vmax=smooth_saliency.max()), cmap='coolwarm', edgecolors='face')
        plt.pcolor(x,y,grid_info, cmap="bw", fc="none", edgecolors=None, lw=0.5)
        plt.title("Base Saliency Map", y=1.01)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("figures/base_saliency_map_{}.pdf".format(grid_choice), bbox_inches='tight')
        print("Figure {} saved".format(grid_choice))
        plt.close()

        img.requires_grad_()
        scores = net_lcs(img)
        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]
        score_max.backward()
        saliency, _ = torch.max(img.grad.data.abs(),dim=1)
        smooth_saliency = gaussian_filter(saliency[0], sigma=1)

        x,y = np.meshgrid(np.arange(grid_info.shape[1]),np.arange(grid_info.shape[0]))
        plt.pcolor(x,y,smooth_saliency,norm=colors.LogNorm(vmin=smooth_saliency.min(), vmax=smooth_saliency.max()), cmap='coolwarm', edgecolors='face')
        plt.pcolor(x,y,grid_info, cmap="bw", fc="none", edgecolors=None, lw=0.5)
        plt.title("LCS Saliency Map", y=1.01)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("figures/lcs_saliency_map_{}.pdf".format(grid_choice), bbox_inches='tight')
        print("Figure {} saved".format(grid_choice))
        plt.close()

        img.requires_grad_()
        scores = net_lcs_p(img)
        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]
        score_max.backward()
        saliency, _ = torch.max(img.grad.data.abs(),dim=1)
        smooth_saliency = gaussian_filter(saliency[0], sigma=1)

        x,y = np.meshgrid(np.arange(grid_info.shape[1]),np.arange(grid_info.shape[0]))
        plt.pcolor(x,y,smooth_saliency,norm=colors.LogNorm(vmin=smooth_saliency.min(), vmax=smooth_saliency.max()), cmap='coolwarm', edgecolors='face')
        plt.pcolor(x,y,grid_info, cmap="bw", fc="none", edgecolors=None, lw=0.5)
        plt.title("LCS-P Saliency Map", y=1.01)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("figures/lcs_p_saliency_map_{}.pdf".format(grid_choice), bbox_inches='tight')
        print("Figure {} saved".format(grid_choice))
        plt.close()

if evolution==True:
    fig, ax = plt.subplots()
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_xticks([])
    ax.set_yticks([])

    smooth_saliency_plot = np.zeros([len(evolution_grid_data), 64, 64])
    evolution_committor = np.zeros([len(evolution_grid_data), 2])

    ising_grid = evolution_grid_data[-3,0]
    grid_x = np.argmax(np.sum(ising_grid, axis=-2))
    grid_y = np.argmax(np.sum(ising_grid, axis=-1))
    evolution_grid_data = np.roll(evolution_grid_data, 32-grid_y, axis=-1)
    evolution_grid_data = np.roll(evolution_grid_data, 32-grid_x, axis=-2)

    for i in range(len(evolution_grid_data)):
        grid_info = evolution_grid_data[i][0]
        img = torch.from_numpy(grid_info.astype(np.float32))
        img = torch.reshape(img, [1,1,64,64])

        img.requires_grad_()
        scores = net_base(img)
        score_max_index = scores.argmax()
        score_max = scores[0,score_max_index]
        score_max.backward()
        saliency, _ = torch.max(img.grad.data.abs(),dim=1)
        smooth_saliency = gaussian_filter(saliency[0], sigma=1)

        smooth_saliency_plot[i] = smooth_saliency
        evolution_committor[i] = [evolution_committor_data[i,0].item(),scores.item()]

    grid_info = evolution_grid_data[0][0]
    smooth_saliency = smooth_saliency_plot[0]
    x,y = np.meshgrid(np.arange(grid_info.shape[1]),np.arange(grid_info.shape[0]))
    heat_map = ax.pcolor(x,y,smooth_saliency,norm=colors.LogNorm(vmin=smooth_saliency.min(), vmax=smooth_saliency.max()), cmap='coolwarm', edgecolors='face')
    lattice = ax.pcolor(x,y,grid_info, cmap="bw", fc="none", edgecolors=None, lw=0.5)
    committor_text = ax.text(0.01, 0.99, 'Committor: {}, Prediction: {}'.format(round(evolution_committor[0,0],4), round(evolution_committor[0,1],4)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top', c='g')

    def animate(i):
        i += 1
        grid_info = evolution_grid_data[i][0]
        smooth_saliency = smooth_saliency_plot[i]
        heat_map.set_array(smooth_saliency.flatten())
        heat_map.set_norm(colors.LogNorm(vmin=smooth_saliency.min(), vmax=smooth_saliency.max()))
        heat_map.autoscale()
        lattice.set_array(grid_info.flatten())
        lattice.autoscale()
        committor_text.set_text('Committor: {}, Prediction: {}'.format(round(evolution_committor[i,0],4), round(evolution_committor[i,1],4)))
        return lattice, heat_map, committor_text

    anim = animation.FuncAnimation(fig=fig, func=animate, interval=500, blit=True, frames=(len(evolution_grid_data)-1), repeat=False)
    #writergif = animation.PillowWriter(fps=1) 
    #anim.save('figures/evolution.gif', writer=writergif)
    writer = animation.FFMpegWriter(fps=4)
    anim.save('figures/evolution.mp4', writer = writer, dpi=360)
    plt.close()
    #plt.show()