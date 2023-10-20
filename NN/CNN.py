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
from torch.optim.lr_scheduler import StepLR
from math import log10, floor
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap

# finding exponent of number
def find_exp(number) -> int:
    base10 = log10(abs(number))
    return floor(base10)

# selection index for labels training
selection = 7

# 0) Load data
# Defining class for loading data
from nn_class import IsingDataset

# Defining function for calculating accuracy
from nn_class import test_accuracy

# Defining function for loading data and transformations
from nn_class import load_data

# 1) Model
from nn_class import ConvNet
    
# Device configuration and setting up network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

#net = ConvNet().to(device)
#net_cpu = ConvNet().to(device_cpu)  

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
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# Loading data
trainset, testset, test_size = load_data()
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=2, batch_size=train_batch_size)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2, batch_size=test_batch_size)

# 3) Training loop

if selection == 0:
    PATH = './model_base.pth'
elif selection == 3:
    PATH = './model_lcs.pth'
elif selection == 7:
    PATH = './model_lcs_p.pth'

# Initializing counters
accuracy = 0
revert_to_checkpoint = 0
revert_limit = 0
revert_break = 0
epoch = 0
loss_offset = 0
min_loss = np.inf
lr_power = 0

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
    net.load_state_dict(checkpoint_model['model_state_dict'])
    optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
    epoch = checkpoint_model['epoch']
    accuracy = checkpoint_model['accuracy']
    loss_arr = np.load("loss.npy")
    net.eval()
    print("Loaded NN")

# Running training loop
run = 1
if run == 1:
    print('Beginning Training Loop')
    n_total_steps = np.int32(len(trainset)/train_batch_size)
    correct = 0
    while (1):
        epoch += 1
        if epoch > num_epochs or revert_break >= 3:
            print("Last epoch reached or revertion break threshold reached.")
            break
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            # index of labels picks what to train on
            labels = labels[:,selection].to(device)
            outputs = torch.flatten(net(images))
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        revert_to_checkpoint += 1 # Incrementing checkpoint counter
        test_loss = loss.item()
        loss_arr[epoch-1] = test_loss
        if selection == 0:
            np.save("loss_base.npy", loss_arr)
        elif selection == 3:
            np.save("loss_lcs.npy", loss_arr)
        elif selection == 7:
            np.save("loss_lcs_p.npy", loss_arr)
        # Test loss for reset
        if test_loss > 0.1 and epoch == 20 and reset == 1: # Resetting optimizer if accuracy is too low
            epoch = 1
            net = ConvNet().to(device)
            print("Loss too high, resetting network and restarting from epoch 1.")
        # Saving checkpoint
        if (min_loss > test_loss):
            revert_to_checkpoint = 0
            revert_limit = 0
            revert_break = 0
            min_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                }, PATH)
        # If no improvement after x epochs, revert to previous checkpoint
        if revert_to_checkpoint == 10 and checkpoint == 1 and epoch > 20:
            revert_to_checkpoint = 0
            revert_limit += 1
            lr_power += 1
            if revert_limit > 3 and checkpoint == 1: # Reset learning rate if no improvement after 10 checkpoint revertions
                revert_limit = 0
                lr_power = 0
                learning_rate = 1.01*learning_rate
                revert_break += 1
                checkpoint_model = torch.load(PATH)
                net.load_state_dict(checkpoint_model['model_state_dict'])
                optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
                epoch = checkpoint_model['epoch']
                optimizer.param_groups[0]["lr"] = learning_rate
                print("Revertion limit reached, resetting learning rate to adjusted initial value.")
                continue
            checkpoint_model = torch.load(PATH)
            net.load_state_dict(checkpoint_model['model_state_dict'])
            optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
            epoch = checkpoint_model['epoch']
            optimizer.param_groups[0]["lr"] = learning_rate*0.5**lr_power
            print("Checkpoint loaded. Starting from epoch: ", epoch)
        lr = optimizer.param_groups[0]["lr"]
        print (f'Epoch [{epoch}/{num_epochs}], Loss: {test_loss:.8f}, Minimum Loss: {min_loss:.6f}, Learning Rate: {lr:.10f}')


test_batch_size = 1
outputs_labels = np.zeros([test_size, 11])
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2, batch_size=test_batch_size)
net_cpu.load_state_dict(net.state_dict())
net_cpu.eval()
_, outputs_labels = test_accuracy(net_cpu, testloader, output_label=outputs_labels)

if selection == 0:
    np.save("prediction_actual_base.npy", outputs_labels)
elif selection == 3:
    np.save("prediction_actual_lcs.npy", outputs_labels)
elif selection == 7:
    np.save("prediction_actual_lcs_p.npy", outputs_labels)