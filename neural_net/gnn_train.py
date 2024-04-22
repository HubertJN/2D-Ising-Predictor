import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

# negative log likelihood
def nll(y, alpha, beta):
    dist = torch.distributions.beta.Beta(alpha, beta)
    neg_log_like = torch.mean(torch.clamp(-dist.log_prob(y), max=10))
    return neg_log_like

run = True

# 0) Load data
# Importing function to load data
from modules.load_data_gnn import load_data

# Importing function for accuracy testing
from modules.test_accuracy import test_accuracy

# 1) Model
from modules.graph_net import graph_net

# Device configuration and setting up network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

k_edge = 8
hidden_n = 8
hidden_m = 512
net = graph_net(k_edge=k_edge, hidden_n=hidden_n, hidden_m=hidden_m).to(device)

# Printing number of parameters
total_params = sum(p.numel() for p in net.parameters())
print("Parameters: ", total_params)

# 2) Hyper-parameters, loss and optimizer, data loading
num_epochs = 10000
learning_rate = 1e-3
weight_decay = 1e-4
train_batch_size = 64

# Loss and optimizer
criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.75)

# Loading data
trainset, valset, testset, val_size, test_size = load_data()
trainloader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size)
valloader = DataLoader(valset, shuffle=False)
testloader = DataLoader(testset, shuffle=False)

# Initialise weights
def weights_init(m):
    if isinstance(m, (nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.0)

net.apply(weights_init)

# 3) Training loop

# Initializing counters
accuracy = 0
revert_to_checkpoint = 0
revert_limit = 0
revert_break = 0
epoch = 0
min_loss = np.inf
global_min_loss = np.inf
lr_power = 0
num_model = 0

# storage for loss
train_loss_arr = np.zeros(num_epochs)
val_loss_arr = np.zeros(num_epochs)
# Running training loop

if run == True:
    print("Beginning Training Loop")
    while (1):
        if epoch >= num_epochs:
            break

        epoch += 1

        cum_num = 0
        cum_loss = 0

        net.train()
        for i, batch in enumerate(trainloader):
            features = batch.x
            edge_index = batch.edge_index
            labels = batch.y

            features = features.to(device)
            edge_index = edge_index.to(device)
            
            # index of labels picks what to train on
            labels = labels.to(device)
            outputs = net(features, edge_index, 4096)
            loss = nll(labels, outputs[:,0], outputs[:,1])
            cum_loss += loss.item()
            cum_num += 1
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        train_loss = cum_loss/cum_num
        train_loss_arr[epoch-1] = train_loss

        cum_num = 0
        cum_loss = 0

        net.eval()
        for i, batch in enumerate(valloader):
            features = batch.x
            edge_index = batch.edge_index
            labels = batch.y

            features = features.to(device)
            edge_index = edge_index.to(device)

            # index of labels picks what to train on
            labels = labels.to(device)
            outputs = net(features, edge_index, 4096)
            loss = nll(labels, outputs[:,0], outputs[:,1])
            cum_loss += loss.item()
            cum_num += 1

        val_loss = cum_loss/cum_num
        val_loss_arr[epoch-1] = val_loss

        if epoch%10 == 0:
            print (f"Epoch [{epoch:05d}/{num_epochs:05d}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
        

PATH = "./models/model_gnn.pth"

torch.save({
    "model_state_dict": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, PATH)

gnn_plot_data = np.zeros([test_size, 3])
net.eval()
for i, batch in enumerate(testloader):
    features = batch.x
    edge_index = batch.edge_index
    labels = batch.y

    features = features.to(device)
    edge_index = edge_index.to(device)
            
    # index of labels picks what to train on
    labels = labels.to(device)
    outputs = net(features, edge_index, 4096)

    gnn_plot_data[i,0] = labels.item()
    gnn_plot_data[i,1] = outputs[0,0].item()
    gnn_plot_data[i,2] = outputs[0,1].item()

#gnn_plot_data = np.zeros([test_size, 2])
#for i, batch in enumerate(testloader):
#    features = batch.x
#    edge_index = batch.edge_index
#    labels = batch.y
#
#    features = features.to(device)
#    edge_index = edge_index.to(device)
#            
#    # index of labels picks what to train on
#    labels = labels.to(device)
#    outputs = net(features, edge_index, 4096)
#    loss = criterion(outputs, labels)
#
#    for j in range(labels.shape[0]):
#        gnn_plot_data[i*test_batch_size+j,0] = labels[j].item()
#        gnn_plot_data[i*test_batch_size+j,1] = outputs[j].item()        

#gnn_plot_data[:,1] = np.clip(gnn_plot_data[:,1], 0, 1)

np.save("./plotting_data/prediction_actual_gnn.npy", gnn_plot_data)
np.save("./plotting_data/train_loss_gnn.npy", train_loss_arr)
np.save("./plotting_data/val_loss_gnn.npy", val_loss_arr)
