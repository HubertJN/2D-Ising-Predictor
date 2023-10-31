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

# selection index for labels training: 0, 3, 7
selection = 0

# 0) Load data
# Defining class for loading data
from nn_module import IsingDataset

# Defining function for calculating accuracy
from nn_module import test_accuracy

# Defining function for loading data and transformations
from nn_module import load_data

# 1) Model
from nn_module import ConvNet
    
# Device configuration and setting up network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

net = ConvNet().to(device)
net_cpu = ConvNet().to(device_cpu)  

# Printing number of parameters
total_params = sum(p.numel() for p in net.parameters())
print("Parameters: ", total_params)
#exit()

# 2) Hyper-parameters, loss and optimizer, data loading
num_epochs = 100
learning_rate = 1e-3
weight_decay = 0
train_batch_size = 100
test_batch_size = 10

# Loss and optimizer
criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Loading data
trainset, testset, test_size = load_data()
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=2, batch_size=train_batch_size)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2, batch_size=test_batch_size)

# Initialise weights
def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight.data)

net.apply(weights_init)

# 3) Training loop

if selection == 0:
    PATH = "./models/model_base.pth"
    loss_path = "./plotting_data/loss_base.npy"
    prediction_actual_path = "./plotting_data/prediction_actual_base.npy"
elif selection == 3:
    PATH = "./model_lcs.pth"
    loss_path = "./plotting_data/loss_lcs.npy"
    prediction_actual_path = "./plotting_data/prediction_actual_lcs.npy"
elif selection == 7:
    PATH = "./models/model_lcs_p.pth"
    loss_path = "./plotting_data/loss_lcs_p.npy"
    prediction_actual_path = "./plotting_data/prediction_actual_lcs_p.npy"

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

# checkpoint system
checkpoint = 1

# reset system
reset = 1

# storage for loss
loss_arr = np.zeros(num_epochs)

# Loading model
load = 0
if load == 1:
    checkpoint_model = torch.load(PATH)
    net.load_state_dict(checkpoint_model["model_state_dict"])
    optimizer.load_state_dict(checkpoint_model["optimizer_state_dict"])
    epoch = checkpoint_model["epoch"]
    accuracy = checkpoint_model["accuracy"]
    loss_arr = np.load(loss_path)
    net.eval()
    print("Loaded NN")

# Running training loop
models = 1
run = 1
if run == 1:
    print("Beginning Training Loop")
    while (1):
        if epoch > num_epochs or revert_break >= 4:
            if (global_min_loss > min_loss):
                global_min_loss = min_loss
                np.save(loss_path, loss_arr[:epoch])
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                    }, PATH)
            epoch = 0
            revert_break = 0
            num_model += 1
            lr_power = 0
            if num_model < models:
                print("Training new model")
            net.apply(initialize_weights)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            loss_arr = np.zeros(num_epochs)
            min_loss = np.inf

        if num_model >= models:
            print("Training finished")
            print("{} models trained".format(models))
            print(f"Minimum loss of best model: {global_min_loss:.6f}")
            break

        epoch += 1

        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            # index of labels picks what to train on
            labels = labels[:,selection].to(device)
            outputs = torch.flatten(net(images, 1))
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        revert_to_checkpoint += 1 # Incrementing checkpoint counter
        test_loss = loss.item()
        loss_arr[epoch-1] = test_loss

        # Test loss for reset
        if test_loss > 0.1 and epoch == 5 and reset == 1: # Resetting optimizer if accuracy is too low
            epoch = 1
            net.apply(initialize_weights)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            print("Network not converging")
            print("Re-initialising weights and restarting training")

        # Saving checkpoint
        if (min_loss > test_loss):
            revert_to_checkpoint = 0
            revert_limit = 0
            revert_break = 0
            min_loss = test_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": accuracy,
                }, "./models/training.pth")

        lr = optimizer.param_groups[0]["lr"]
        print (f"Epoch [{epoch:03d}/{num_epochs}], Loss: {test_loss:.8f}, Minimum Loss: {min_loss:.6f}, Learning Rate: {lr:.10f}")
        
        # If no improvement after x epochs, revert to previous checkpoint
        if revert_to_checkpoint == 10 and checkpoint == 1 and epoch > 5:
            revert_to_checkpoint = 0
            revert_limit += 1
            revert_break += 1
            lr_power += 1

            checkpoint_model = torch.load("./models/training.pth")
            net.load_state_dict(checkpoint_model["model_state_dict"])
            optimizer.load_state_dict(checkpoint_model["optimizer_state_dict"])
            epoch = checkpoint_model["epoch"]
            optimizer.param_groups[0]["lr"] = learning_rate*0.5**lr_power
            if (revert_break < 4):
                print("Checkpoint loaded. Starting from epoch: ", epoch)


test_batch_size = 1
outputs_labels = np.zeros([test_size, 11])
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2, batch_size=test_batch_size)

best_model = torch.load(PATH)
net_cpu.load_state_dict(best_model["model_state_dict"])

net_cpu.eval()
_, outputs_labels = test_accuracy(net_cpu, testloader, selection, test_batch_size, output_label=outputs_labels)

np.save(prediction_actual_path, outputs_labels)