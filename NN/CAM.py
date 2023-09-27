import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams['figure.dpi'] = 120

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

committor_data = committor_data.numpy()
grid_data = grid_data.numpy()

# Ising grid
cluster_choice = 200
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
    plt.imshow(grid_info, cmap='Greys_r')
    plt.imshow(CAMs, cmap='alpha')
    plt.title("Class Activation Map", y=1.01)
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
    plt.imshow(grid_info, cmap='Greys_r')
    plt.imshow(CAMs, cmap='alpha')
    plt.title("Class Activation Map", y=1.01)
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
    plt.imshow(grid_info, cmap='Greys_r')
    plt.imshow(CAMs, cmap='alpha')
    plt.title("Class Activation Map", y=1.01)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig("figures/lcs_p_class_activation_map_{}.pdf".format(grid_choice), bbox_inches='tight')
    print("Figure {} saved".format(grid_choice))
    plt.close()
