"""
============================================================================================
                                 cnn_training.py

Python file containing convolutional neural network training loop.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch
import numpy as np
import time

def net_training(start_epoch, epochs, total_epochs, net, device, loss_func, optimizer, scheduler, train_loader, val_loader, train_loss_arr, val_loss_arr):
    """net_training
    Training loops for graph neural network

    Parameters:
    epochs: number of training epochs
    net: neural network object
    device: device to which pytorch tensors are loaded
    loss_func: loss function
    optimizer: PyTorch optimizer object
    scheduler: PyTroch scheduler object
    train_loader: training data dataloader
    val_loader: validation data dataloader

    Returns:
    net: trained neural network object
    train_loss_arr: training loss array
    val_loss_arr: validation loss array
    time_taken: time taken for training loop
    """
    begin = time.time()

    for epoch in range(start_epoch, start_epoch+epochs):
        cum_num = 0
        cum_loss = 0

        # training loop
        net.train()
        for i, (features, labels, _) in enumerate(train_loader):
            # sending data to device
            features = features.to(device)
            labels = labels.to(device)

            # performing forward pass
            predictions = net(features)

            # calculating loss
            loss = loss_func(labels, predictions[:,0], predictions[:,1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cumulative loss across batch for terminal display
            cum_loss += loss.item()
            cum_num += 1

        train_loss = cum_loss/cum_num
        train_loss_arr[epoch-1] = train_loss

        cum_num = 0
        cum_loss = 0

        # validation loop
        net.eval()
        for i, (features, labels, _) in enumerate(val_loader):
            # sending data to device
            features = features.to(device)
            labels = labels.to(device)

            # performing forward pass
            predictions = net(features)

            # calculating loss
            loss = loss_func(labels, predictions[:,0], predictions[:,1])

            # cumulative loss across batch for terminal display
            cum_loss += loss.item()
            cum_num += 1

        val_loss = cum_loss/cum_num
        val_loss_arr[epoch-1] = val_loss
        
        if epoch%10 == 0 or epoch == 1:
            end = time.time()
            print (f"Epoch [{epoch:05d}/{total_epochs:05d}], Train Loss: {train_loss:+.8f}, Val Loss: {val_loss:+.8f}, Time Taken: {end-begin:.1f} seconds, Learning Rate: {scheduler.get_last_lr()[0]:.2E}")
            begin = time.time()
            scheduler.step()
            
    return net, train_loss_arr, val_loss_arr
        
