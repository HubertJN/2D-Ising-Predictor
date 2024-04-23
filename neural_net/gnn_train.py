import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)
from torch_geometric.loader import DataLoader

# 1) loading modules
##################################################
# import data loading function
from modules.gnn_load_data import load_data

# import model architecture
from modules.gnn_architecture import graph_net

# import training function
from modules.gnn_training import gnn_training

# import weight initialization
from modules.gnn_weight_initialization import weights_init

# import weight initialization
from modules.nll_loss_function import loss_func

# 2) setup
##################################################
# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network hyper-parameter choice and initialization
k_edge = 8
hidden_n = 8
hidden_m = 512
net = graph_net(k_edge=k_edge, hidden_n=hidden_n, hidden_m=hidden_m).to(device)
net.apply(weights_init) # initialise weights

# training hyper-parameters
epochs = 5000 # number of training cycles over data
learning_rate = 5e-4
weight_decay = 1e-4 # weight parameter for L2 regularization
train_batch_size = 64
scheduler_step = 250 # steps before scheduler_gamma is applied to learning rate
scheduler_gamma = 0.75 # learning rate multiplier every scheduler_step epochs

# optimizer and scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# loading data
trainset, valset, testset, train_size, val_size, test_size = load_data()
train_loader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size)
val_loader = DataLoader(valset, shuffle=False)
test_loader = DataLoader(testset, shuffle=False)

# 3) training loop
##################################################
# printing number of parameters
total_params = sum(p.numel() for p in net.parameters())
print("Parameters: ", total_params)

# running training loop
net, train_loss, val_loss = gnn_training(epochs, net, device, loss_func, optimizer, scheduler, train_loader, val_loader) 

# 4) saving and plotting data output
##################################################
PATH = "./models/model_gnn.pth"
torch.save({
    "model_state_dict": net.state_dict(),
    }, PATH)

gnn_plot_data = np.zeros([test_size, 3])
net.eval()
for i, batch in enumerate(test_loader):
    features = batch.x
    edge_index = batch.edge_index
    labels = batch.y

    features = features.to(device)
    edge_index = edge_index.to(device)
            
    # index of labels picks what to train on
    labels = labels.to(device)
    outputs = net(features, edge_index, len(labels))

    gnn_plot_data[i,0] = labels.item()
    gnn_plot_data[i,1] = outputs[0,0].item()
    gnn_plot_data[i,2] = outputs[0,1].item()

np.save("./plotting_data/prediction_actual_gnn.npy", gnn_plot_data)
np.save("./plotting_data/train_loss_gnn.npy", train_loss)
np.save("./plotting_data/val_loss_gnn.npy", val_loss)
