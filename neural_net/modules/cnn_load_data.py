"""
============================================================================================
                                 cnn_load_data.py

Python file containing functions for initializing and populating convolutional neural 
network data arrays. Loads data in from external files and splits them for training, 
validation and testing.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

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
    image_dir = "./training_data/image_data_subset"
    label_dir = "./training_data/label_data_subset"
    image_data = torch.load(image_dir)
    label_data = torch.load(label_dir)[:,0]
    id_data = torch.arange(len(image_data))

    image_train, image_test, label_train, label_test, id_train, id_test = train_test_split(image_data, label_data, id_data, test_size=0.2)
    image_train, image_val, label_train, label_val, id_train, id_val = train_test_split(image_train, label_train, id_train, test_size=0.2)

    trainset = ising_dataset(image_train, label_train, id_train, device, train=True)
    valset = ising_dataset(image_val, label_val, id_val, device)
    testset = ising_dataset(image_test, label_test, id_test, device)

    train_size = len(trainset)
    val_size = len(valset)
    test_size = len(testset)

    return trainset, valset, testset, train_size, val_size, test_size

class ising_dataset(Dataset):
    def __init__(self, data, labels, index, device="cpu", train=False):
        """__init__
        Initializes Dataset object for Ising data

        Parameters:
        data: image array of Ising grids
        labels: committor values for each image
        index: index of graph within whole data set
        device: device to which pytorch tensors are loaded (default is cpu)
        train: whether data set is used for training or not (default is False)

        Returns:
        None
        """
        self.img_labels = labels.to(device)
        self.img_data = data.to(device)
        self.train = train
        self.index = index

    def __len__(self):
        """__len__
        Returns length of data set

        Parameters:
        None

        Returns:
        len(self.labels): length of data set
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """__getitem__
        Gets individual item from data set

        Parameters:
        idx: id of item to be extracted from data set

        Returns:
        image: Ising grid image (transformed if train parameter True in __init__)
        label: committor value
        index: index of item within whole data set
        """
        image = self.img_data[idx]
        label = self.img_labels[idx]
        index = self.index[idx]
        if self.train==True:
            #image = transforms.functional.rotate(image, 90)/4 + transforms.functional.rotate(image, 180)/4 + transforms.functional.rotate(image, 270)/4 + image/4
            hflipper = transforms.RandomHorizontalFlip(p=0.5)
            vflipper = transforms.RandomVerticalFlip(p=0.5)
            image = transforms.functional.rotate(image, np.random.randint(4)*90)
            image = hflipper(image)
            image = vflipper(image)
            image = torch.roll(image, shifts=(np.random.randint(image.shape[-1]),image.shape[-2]), dims=(-1, -2))
        return image, label, index
