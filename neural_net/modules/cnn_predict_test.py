import torch
import torch.nn as nn

def predict_test(net, test_loader, plot_data):
    for i, (features, labels, index) in enumerate(test_loader):
        features = features
        labels = labels

        predictions = net(features)

        plot_data[i,0] = labels.item()
        plot_data[i,1] = predictions[0,0].item()
        plot_data[i,2] = predictions[0,1].item()
        plot_data[i,3] = index.item()

    return plot_data
