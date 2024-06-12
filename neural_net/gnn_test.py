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
if len(sys.argv) > 1: # if command line arguments provided
        run = int(sys.argv[1])
        k_edge = int(sys.argv[2])
        hidden_n = int(sys.argv[3])
else:
    print("Error. No input values.")
    print("gnn_test.py run k_edge hidden_n")
    exit()

# loading data
trainset, valset, testset, train_size, val_size, test_size = load_data(device)
test_loader = DataLoader(testset, shuffle=False)

net = graph_net(k_edge=k_edge, hidden_n=hidden_n).to(device)

PATH = "./models/gnn_model_%d.pth" % run
net.load_state_dict(torch.load(PATH)['model_state_dict'])
net.eval()

plot_data = np.zeros([test_size, 4])
for i, (batch, index) in enumerate(test_loader):
    features = batch.x
    edge_index = batch.edge_index
    labels = batch.y

    # index of labels picks what to train on
    predictions = net(features, edge_index, len(labels))

    plot_data[i,0] = labels.item()
    plot_data[i,1] = predictions[0,0].item()
    plot_data[i,2] = predictions[0,1].item()
    plot_data[i,3] = index.item()

np.save("./plotting_data/gnn/gnn_prediction_actual_%d.npy" % run, plot_data)

alpha = plot_data[:,1]
beta = plot_data[:,2]
expectation = alpha/(alpha+beta)
rmse = np.sqrt(np.mean((plot_data[:,0]-expectation)**2))
print("RMSE = %f" % rmse)
