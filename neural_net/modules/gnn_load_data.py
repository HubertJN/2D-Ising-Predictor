import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from torch_geometric.data import Data

def load_data(device):  
    image_dir = "./training/data/image_data_subset"
    feature_dir = "./training_data/feature_data_subset"
    edge_dir = "./training_data/edge_data_subset"
    label_dir = "./training_data/label_data_subset"

    feature_data = torch.load(feature_dir)
    edge_data = torch.load(edge_dir) 
    label_data = torch.load(label_dir)
    id_data = torch.arange(len(feature_data))

    feature_train, feature_test, edge_train, edge_test, label_train, label_test, id_train, id_test = train_test_split(feature_data, edge_data, label_data, id_data, test_size=0.2)
    feature_train, feature_val, edge_train, edge_val, label_train, label_val, id_train, id_val = train_test_split(feature_train, edge_train, label_train, id_train, test_size=0.1)

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
        self.features = features.to(device)
        self.edge = edge.to(device)
        self.index = index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        edge = self.edge[idx]
        labels = self.labels[idx]
        index = self.index[idx]
        
        # GCN data type
        graph = Data(x=features, edge_index=edge, y=labels)
        return graph, index
