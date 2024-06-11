import torch
import torch.nn as nn

def predict_test(net, test_loader, plot_data):
    for i, (batch, index) in enumerate(test_loader):
        features = batch.x
        edge_index = batch.edge_index
        labels = batch.y

        predictions = net(features, edge_index, len(labels))

    plot_data[i,0] = labels.item()
    plot_data[i,1] = predictions[0,0].item()
    plot_data[i,2] = predictions[0,1].item()
    plot_data[i,3] = index.item()

    return plot_data
