"""
============================================================================================
                                 cnn_predict_test.py

Python file containing loss function for testing convolutional neural network performance.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch
import torch.nn as nn

def predict_test(net, data_loader, plot_data):
    """predict_test
    Given network and data_loader populates array with data required for plotting

    Parameters:
    net: neural network object
    data_loader: pytorch data_loader
    plot_data: array for data storage

    Returns:
    plot_data: filled array (label, alpha, beta, index)
    """
    for i, (features, labels, index) in enumerate(data_loader):
        features = features
        labels = labels

        predictions = net(features)

        plot_data[i,0] = labels.item()
        plot_data[i,1] = predictions[0,0].item()
        plot_data[i,2] = predictions[0,1].item()
        plot_data[i,3] = index.item()

    return plot_data
