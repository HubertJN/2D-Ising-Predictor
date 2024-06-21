"""
============================================================================================
                                 gnn_load_data.py

Python file containing functions for initializing and populating graph neural network data
arrays. Loads data in from external files and splits them for training, validation and
testing.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from torch_geometric.data import Data

def load_data(device):
    """load_data
    Loads data from files then splits them for training, validation and testing.

    Parameters:
    device: device to which pytorch tensors are loaded

    Returns:
    trainset: training data set
    valset: validation data set
    testset: testing data set
    train_size: size of training data set
    val_size: size of validation data set
    test_size: size of testing data set
    """
    image_dir = "./training/data/image_data_subset"
    feature_dir = "./training_data/feature_data_subset"
    edge_dir = "./training_data/edge_data_subset"
    label_dir = "./training_data/label_data_subset"

    feature_data = torch.load(feature_dir)
    edge_data = torch.load(edge_dir) 
    label_data = torch.load(label_dir)[:,0]
    id_data = torch.arange(len(feature_data))

    feature_train, feature_test, edge_train, edge_test, label_train, label_test, id_train, id_test = train_test_split(feature_data, edge_data, label_data, id_data, test_size=0.2, random_state=1)
    feature_train, feature_val, edge_train, edge_val, label_train, label_val, id_train, id_val = train_test_split(feature_train, edge_train, label_train, id_train, test_size=0.1, random_state=1)

    trainset = ising_dataset(feature_train, edge_train, label_train, id_train, device)
    valset = ising_dataset(feature_val, edge_val, label_val, id_val, device)
    testset = ising_dataset(feature_test, edge_test, label_test, id_test, device)

    train_size = len(trainset)
    val_size = len(valset)
    test_size = len(testset)

    return trainset, valset, testset, train_size, val_size, test_size

class ising_dataset(Dataset):
    def __init__(self, features, edge, labels, index, device="cpu"):
        self.labels = labels.to(device)
        """__init__
        Initializes Dataset object for Ising data

        Parameters:
        features: feature data of ising grid
        edge: edge data of graph
        labels: committor values for each graph
        index: index of graph within whole data set
        device: device to which pytorch tensors are loaded (default is cpu)

        Returns:
        None
        """
        # normal dist blurring of features
        #features = torch.normal(mean=features, std=0.005)
        
        # random translation of features
        #shape = features.shape
        #features = torch.reshape(features, (-1,64,64)); test = features
        #features = torch.roll(features, shifts=(np.random.randint(features.shape[-1]),features.shape[-2]), dims=(-1, -2))
        #features = torch.reshape(features, shape)

        self.features = features.to(device)
        self.edge = edge.to(device)
        self.index = index

    def __len__(self):
        """__len__
        Returns length of data set

        Parameters:
        None

        Returns:
        len(self.labels): length of data set
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """__getitem__
        Gets individual item from data set

        Parameters:
        idx: id of item to be extracted from data set

        Returns:
        graph: graph output as a torch_geometric object
        index: index of item within whole data set
        """
        features = self.features[idx]
        edge = self.edge[idx]
        labels = self.labels[idx]
        index = self.index[idx]
        
        # GCN data type
        graph = Data(x=features, edge_index=edge, y=labels)
        return graph, index
