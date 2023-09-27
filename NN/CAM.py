import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams['figure.dpi'] = 120

# plot or evolution?
plot = False
evolution = True

# 1) Model

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # fully connected layers
        self.fc1 = nn.Linear(16*16*16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        # pooling
        self.pool = nn.MaxPool2d(2,2)

        # convolution layers
        self.conv1 = nn.Conv2d(1, 16, 7, padding=[3,3], padding_mode="circular") # Maybe pad to 10 cause area of cluster
        self.conv2 = nn.Conv2d(16, 16, 5, padding=[2,2], padding_mode="circular")
        self.conv3 = nn.Conv2d(16, 16, 3, padding=[1,1], padding_mode="circular")

        # drop out
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        # layer 1
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        # layer 2
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        # layer 3
        x = F.leaky_relu(self.conv3(x))

        # flatten
        x = x.view(-1, 16*16*16)
        
        # fully connected layer
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x
    
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

# Setup class activation map
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net_base._modules.get("conv3").register_forward_hook(hook_feature)
net_lcs._modules.get("conv3").register_forward_hook(hook_feature)
net_lcs_p._modules.get("conv3").register_forward_hook(hook_feature)

# get colormap
ncolors = 256
color_array = np.zeros([ncolors, 4])

# change alpha values
color_array[:,-1] = np.linspace(1.0,0.0, ncolors)
color_array[:,0] = 1
color_array[:,1] = 0
color_array[:,2] = 0

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name='alpha',colors=color_array)
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
    cluster_choice = 250
    for grid_choice in range(3):
        grid_index = np.where(committor_data[:,-1] == cluster_choice)
        grid_info = grid_data[grid_index][grid_choice,0]
        img = torch.from_numpy(grid_info.astype(np.float32))
        img = torch.reshape(img, [1,1,64,64])

        features_blobs = []
        logit = net_base(img)
        features_blobs = np.array(features_blobs)
        CAMs = np.sum(features_blobs[0,0], axis=0)
        CAMs = zoom(CAMs, 64/CAMs.shape[-1])
        # scale CAM to start at 0 and be positive
        CAMs = CAMs - np.min(CAMs)
        plt.imshow(grid_info, cmap='Greys_r')
        plt.imshow(CAMs, cmap='alpha')
        plt.title("Base Class Activation Map", y=1.01)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("figures/base_class_activation_map_{}.pdf".format(grid_choice), bbox_inches='tight')
        print("Figure {} saved".format(grid_choice))
        plt.close()

        features_blobs = []
        logit = net_lcs(img)
        features_blobs = np.array(features_blobs)
        CAMs = np.sum(features_blobs[0,0], axis=0)
        CAMs = zoom(CAMs, 64/CAMs.shape[-1])
        CAMs = CAMs - np.min(CAMs)
        plt.imshow(grid_info, cmap='Greys_r')
        plt.imshow(CAMs, cmap='alpha')
        plt.title("LCS Class Activation Map", y=1.01)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("figures/lcs_class_activation_map_{}.pdf".format(grid_choice), bbox_inches='tight')
        print("Figure {} saved".format(grid_choice))
        plt.close()

        features_blobs = []
        logit = net_lcs_p(img)
        features_blobs = np.array(features_blobs)
        CAMs = np.sum(features_blobs[0,0], axis=0)
        CAMs = zoom(CAMs, 64/CAMs.shape[-1])
        CAMs = CAMs - np.min(CAMs)
        plt.imshow(grid_info, cmap='Greys_r')
        plt.imshow(CAMs, cmap='alpha')
        plt.title("LCS-P Class Activation Map", y=1.01)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("figures/lcs_p_class_activation_map_{}.pdf".format(grid_choice), bbox_inches='tight')
        print("Figure {} saved".format(grid_choice))
        plt.close()

if evolution==True:
    fig, ax = plt.subplots()
    fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_xticks([])
    ax.set_yticks([])

    
    CAM_plot = np.zeros([len(evolution_grid_data), 64, 64])
    evolution_committor = np.zeros([len(evolution_grid_data), 2])

    for i in range(len(evolution_grid_data)):
        grid_info = evolution_grid_data[i][0]
        img = torch.from_numpy(grid_info.astype(np.float32))
        img = torch.reshape(img, [1,1,64,64])

        features_blobs = []
        logit = net_base(img)
        evolution_committor[i] = [evolution_committor_data[i,0].item(),logit.item()]
        features_blobs = np.array(features_blobs)
        CAMs = np.sum(features_blobs[0,0], axis=0)
        CAMs = zoom(CAMs, 64/CAMs.shape[-1])
        CAM_plot[i] = CAMs

    ising_grid = evolution_grid_data[-3,0]
    grid_x = np.argmax(np.sum(ising_grid, axis=0))
    grid_y = np.argmax(np.sum(ising_grid, axis=1))
    evolution_grid_data = np.roll(evolution_grid_data, 32-grid_y, axis=-1)
    evolution_grid_data = np.roll(evolution_grid_data, 32-grid_x, axis=-2)

    grid_info = evolution_grid_data[0][0]
    CAMs = CAM_plot[0]
    lattice = ax.imshow(grid_info, cmap='Greys_r', animated=True)
    heat_map = ax.imshow(CAMs, cmap='alpha', animated=True)
    committor_text = ax.text(0.01, 0.99, 'Committor: {}, Prediction: {}'.format(round(evolution_committor[0,0],4), round(evolution_committor[0,1],4)), transform = ax.transAxes, horizontalalignment='left',
     verticalalignment='top', c='g')

    def animate(i):
        i += 1
        grid_info = evolution_grid_data[i][0]
        CAMs = CAM_plot[i]
        lattice.set_data(grid_info)
        lattice.autoscale()
        heat_map.set_data(CAMs)
        heat_map.autoscale()
        committor_text.set_text('Committor: {}, Prediction: {}'.format(round(evolution_committor[i,0],4), round(evolution_committor[i,1],4)))
        return lattice, heat_map, committor_text

    anim = animation.FuncAnimation(fig=fig, func=animate, interval=500, blit=True, frames=(len(evolution_grid_data)-1), repeat=False)
    #writergif = animation.PillowWriter(fps=1) 
    #anim.save('figures/evolution.gif', writer=writergif)
    writer = animation.FFMpegWriter(fps=4)
    anim.save('figures/evolution.mp4', writer = writer, dpi=240)
    plt.close()
    #plt.show()
    
