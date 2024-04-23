import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from torch_geometric.data import Data

def load_data():  
    feature_dir = "./training_data/image_coord_gnn"
    edge_dir = "./training_data/image_neigh_gnn"
    label_dir = "./training_data/label_data_gnn"
    feature_data = torch.load(feature_dir)
    edge_data = torch.load(edge_dir); edge_data = edge_data.type(torch.int64) 
    label_data = torch.load(label_dir)[:,0]
    
    # sort data
    ordering = label_data.argsort()
    label_data = label_data[ordering]
    edge_data = edge_data[ordering]
    feature_data = feature_data[ordering]

    # map features to [0,1]
    feature_data -= feature_data.min()
    feature_data /= feature_data.max()

    tmp_idx = np.zeros(50000)
    select = 50
    select_count = 0
    sample_space = np.linspace(0.0, 1.0, 101)
    j = 0; k = 0

    for i, sample in enumerate(label_data):
        value = sample_space[j]
        if abs(sample.item() - value) < 0.001 and select_count < select:
            select_count += 1
            tmp_idx[k] = i; k += 1
        elif abs(sample.item() - value) > 0.001:
            select_count = 0
            j += 1
   
    tmp_idx = tmp_idx[:k]
    # implement uniformly sampled data across the range of committors

    #feature_train, feature_test, edge_train, edge_test, label_train, label_test = train_test_split(feature_data, edge_data, label_data, test_size=0.2, random_state=1)
    feature_train, feature_test, edge_train, edge_test, label_train, label_test = train_test_split(feature_data[tmp_idx], edge_data[tmp_idx], label_data[tmp_idx], test_size=0.2, random_state=1)
    feature_train, feature_val, edge_train, edge_val, label_train, label_val = train_test_split(feature_train, edge_train, label_train, test_size=0.2, random_state=1)

    trainset = ising_dataset(feature_train, edge_train, label_train)
    valset = ising_dataset(feature_val, edge_val, label_val)
    testset = ising_dataset(feature_test, edge_test, label_test)

    train_size = len(trainset)
    val_size = len(valset)
    test_size = len(testset)

    return trainset, valset, testset, train_size, val_size, test_size

class ising_dataset(Dataset):
    def __init__(self, features, edge, labels):
        self.labels = labels
        self.features = features
        self.edge = edge

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        edge = self.edge[idx]
        labels = self.labels[idx]
        
        # GCN data type
        graph = Data(x=features, edge_index=edge, y=labels)
        return graph
