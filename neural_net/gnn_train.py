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
run = 13
k_edge = 1
hidden_n = 3

if len(sys.argv) > 1: # if command line arguments provided
        run = int(sys.argv[1])
        k_edge = int(sys.argv[2])
        hidden_n = int(sys.argv[3])

net = graph_net(k_edge=k_edge, hidden_n=hidden_n).to(device)
net.apply(weights_init) # initialise weights

try:
    open("figures/gnn_tweaking/hyperparameters_%d.txt" % run, "r")
    print("Run already exists. Change run variable. Exiting.")
    exit_status = True
except:
    exit_status = False
if exit_status == True:
    sys.exit()


# training hyper-parameters
epochs = 3750 # number of training cycles over data
models = 2 # number of models to train and then select best (lowest loss based on mean of last 50 epochs from validation set)
learning_rate = 1e-3
weight_decay = 1e-4 # weight parameter for L2 regularization
train_batch_size = 64
scheduler_step = 1000 # steps before scheduler_gamma is applied to learning rate
scheduler_gamma = 0.5 # learning rate multiplier every scheduler_step epochs

# optimizer and scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# loading data
trainset, valset, testset, train_size, val_size, test_size = load_data(device)
train_loader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size)
val_loader = DataLoader(valset, shuffle=False, batch_size=train_batch_size)
test_loader = DataLoader(testset, shuffle=False)

# 3) training loop
##################################################
# printing size of dataset
print(f"Training size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
# printing number of parameters
total_params = sum(p.numel() for p in net.parameters())
print("Parameters: ", total_params)

# running training loop
gnn_dict = {}
for mod in range(models):
    net, train_loss, val_loss, time_taken = gnn_training(epochs, net, device, loss_func, optimizer, scheduler, train_loader, val_loader) 

    gnn_dict["net_%d" % mod] = net
    gnn_dict["train_loss_%d" % mod] = train_loss
    gnn_dict["val_loss_%d" % mod] = val_loss
    gnn_dict["time_taken_%d" % mod] = time_taken
    
    net.apply(weights_init) # initialise weights
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

min_mod = np.inf
for mod in range(models):
    min_tmp = np.mean(gnn_dict["val_loss_%d" % mod][-50:])
    if min_tmp < min_mod:
        min_mod = min_tmp
        mod_choice = mod

net = gnn_dict["net_%d" % mod_choice]
train_loss = gnn_dict["train_loss_%d" % mod_choice]
val_loss = gnn_dict["val_loss_%d" % mod_choice]
time_taken = gnn_dict["val_loss_%d" % mod_choice]

# 4) saving and plotting data output
##################################################
PATH = "./models/gnn_model_%d.pth" % run
torch.save({
    "model_state_dict": net.state_dict(),
    }, PATH)

plot_data = np.zeros([test_size, 3])
net.eval()
for i, batch in enumerate(test_loader):
    features = batch.x
    edge_index = batch.edge_index
    labels = batch.y

    # index of labels picks what to train on
    predictions = net(features, edge_index, len(labels))

    plot_data[i,0] = labels.item()
    plot_data[i,1] = predictions[0,0].item()
    plot_data[i,2] = predictions[0,1].item()

# save variables to file
hyperparameters={'run' : run,
         'k_edge' : k_edge, 
         'hidden_n' : hidden_n,
         'total parameters' : total_params,
         'epochs' : epochs,
         'learning_rate' : learning_rate,
         'weight_decay' : weight_decay,
         'train_batch_size' : train_batch_size,
         'scheduler_step (epochs)' : scheduler_step,
         'scheduler_gamma' : scheduler_gamma,
         'time_taken (seconds)' : time_taken,
         'dataset' : train_size + val_size + test_size,
         'train_size' : train_size,
         'val_size' : val_size,
         'test_size' : test_size} 
  
with open("figures/gnn_tweaking/hyperparameters_%d.txt" % run, 'w') as f:  
    for key, value in hyperparameters.items():  
        f.write('%s = %s\n' % (key, value))

np.save("./plotting_data/gnn_prediction_actual_%d.npy" % run, plot_data)
np.save("./plotting_data/gnn_train_loss_%d.npy" % run, train_loss)
np.save("./plotting_data/gnn_val_loss_%d.npy" % run, val_loss)
