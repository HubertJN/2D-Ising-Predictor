import torch
import torch.nn as nn
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)
from torch.utils.data import DataLoader

# 1) loading modules
##################################################
# import data loading function
from modules.cnn_load_data import load_data

# import model architecture
from modules.cnn_architecture import conv_net

# import training function
from modules.cnn_training import cnn_training

# import weight initialization
from modules.cnn_weight_initialization import weights_init

# import weight initialization
from modules.nll_loss_function import loss_func

# 2) setup
##################################################
# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# network hyper-parameter choice and initialization
conv = 3
channels = 8
hidden_n = 128
img_dim = 64
net = graph_net(conv=conv, channels=channels, hidden_n=hidden_n, img_dim=img_dim).to(device)
net.apply(weights_init) # initialise weights

# training hyper-parameters
epochs = 5000 # number of training cycles over data
learning_rate = 5e-4
weight_decay = 1e-4 # weight parameter for L2 regularization
train_batch_size = 64
scheduler_step = 250 # steps before scheduler_gamma is applied to learning rate
scheduler_gamma = 0.75 # learning rate multiplier every scheduler_step epochs

# loss, optimizer and scheduler
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
net, train_loss, val_loss = cnn_training(epochs, net, device, loss_func, optimizer, scheduler, train_loader, val_loader) 

# 4) saving and plotting data output
##################################################
PATH = "./models/cnn_model.pth"
torch.save({
    "model_state_dict": net.state_dict(),
    }, PATH)

gnn_plot_data = np.zeros([test_size, 3])
net.eval()
for i, (features, labels) in enumerate(test_loader):
        # sending data to device
        features = features.to(device)
        labels = labels.to(device)

        # performing forward pass
        predictions = net(features)

        # calculating loss
        loss = loss_func(labels, predictions[:,0], predictions[:,1])

    plot_data[i,0] = labels.item()
    plot_data[i,1] = outputs[0,0].item()
    plot_data[i,2] = outputs[0,1].item()

np.save("./plotting_data/cnn_prediction_actual.npy", plot_data)
np.save("./plotting_data/cnn_train_loss.npy", train_loss)
np.save("./plotting_data/cnn_val_loss.npy", val_loss)
