import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)
from torch_geometric.loader import DataLoader
import importlib

if len(sys.argv) == 5: # if command line arguments provided
        net_type = sys.argv[1].lower()
        run = int(sys.argv[2])
        k_edge = int(sys.argv[3])
        hidden_n = int(sys.argv[4])
else:
    print('Error. Incorrect input parameters.\nnn_train.py net_type run k_edge hidden_n\nnet_type = "cnn" or "gnn"\nrun, k_edge, hidden_n = int > 0')
    exit()

# 1) loading modules
##################################################
if net_type == 'cnn':
    from torch.utils.data import DataLoader
if net_type == 'gnn':
    from torch_geometric.nn import GCNConv
    from torch_geometric.loader import DataLoader
# import data loading function
load_data = getattr(importlib.import_module('modules.%s_load_data' % net_type), 'load_data')

# import model architecture
net_init = getattr(importlib.import_module('modules.%s_architecture' % net_type), 'net_init')

# import training function
net_training = getattr(importlib.import_module('modules.%s_training' % net_type), 'net_training')

# import weight initialization
weight_init = getattr(importlib.import_module('modules.%s_weight_init' % net_type), 'weight_init')

# import loss function
loss_func = getattr(importlib.import_module('modules.nll_loss_function'), 'loss_func')

# import prediction testing
predict_test = getattr(importlib.import_module('modules.%s_predict_test' %net_type), 'predict_test')

# 2) setup
##################################################
# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = net_init(k_edge, hidden_n).to(device)
net.apply(weight_init) # initialise weights

try:
    open("plotting_data/gnn/hyperparameters_%d.txt" % run, "r")
    print("Run already exists. Change run variable. Exiting.")
    exit_status = True
except:
    exit_status = False
if exit_status == True:
    """sys.exit()""" ########################## CHANGE LATER ###########################

# training hyper-parameters
epochs = 10 # number of training cycles over data
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
nn_dict = {}
for mod in range(models):
    net, train_loss, val_loss, time_taken = net_training(epochs, net, device, loss_func, optimizer, scheduler, train_loader, val_loader) 

    nn_dict["net_%d" % mod] = net
    nn_dict["train_loss_%d" % mod] = train_loss
    nn_dict["val_loss_%d" % mod] = val_loss
    nn_dict["time_taken_%d" % mod] = time_taken
    
    net.apply(weight_init) # initialise weights
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

min_mod = np.inf
for mod in range(models):
    min_tmp = np.mean(nn_dict["val_loss_%d" % mod][-50:])
    if min_tmp < min_mod:
        min_mod = min_tmp
        mod_choice = mod

net = nn_dict["net_%d" % mod_choice]
train_loss = nn_dict["train_loss_%d" % mod_choice]
val_loss = nn_dict["val_loss_%d" % mod_choice]
time_taken = nn_dict["time_taken_%d" % mod_choice]

# 4) saving and plotting data output
##################################################
PATH = "./models/%s_model_%d.pth" % (net_type, run)
torch.save({
    "model_state_dict": net.state_dict(),
   }, PATH)

plot_data = np.zeros([test_size, 4])
net.eval()
plot_data = predict_test(net, test_loader, plot_data)

# find rmse of net performance
alpha = plot_data[:,1]
beta = plot_data[:,2]
expectation = alpha/(alpha+beta)
rmse = np.sqrt(np.mean((plot_data[:,0]-expectation)**2))

# save variables to file
hyperparameters={'run' : run,
         'net_type': net_type,
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
         'test_size' : test_size,
         'rmse': rmse
         }
  
with open("plotting_data/%s/hyperparameters_%d.txt" % (net_type, run), 'w') as f:  
    for key, value in hyperparameters.items():  
        f.write('%s = %s\n' % (key, value))

np.save("./plotting_data/%s/%s_prediction_actual_%d.npy" % (net_type, net_type, run), plot_data)
np.save("./plotting_data/%s/%s_train_loss_%d.npy" % (net_type, net_type, run), train_loss)
np.save("./plotting_data/%s/%s_val_loss_%d.npy" % (net_type, net_type, run), val_loss)
