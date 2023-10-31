import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
from math import log10, floor

# finding exponent of number
def find_exp(number) -> int:
    base10 = log10(abs(number))
    return floor(base10)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # fully connected layers
        self.fc1 = nn.Linear(16*16*16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        # pooling
        self.pool = nn.MaxPool2d(2,2)

        # convolution layers
        self.conv1 = nn.Conv2d(1, 16, 7, padding=[3,3], padding_mode="circular")
        self.conv2 = nn.Conv2d(16, 16, 5, padding=[2,2], padding_mode="circular")
        self.conv3 = nn.Conv2d(16, 16, 3, padding=[1,1], padding_mode="circular")

        # drop out
        self.drop = nn.Dropout(p=0.1)

        # resize
        self.resize = transforms.Resize((64, 64), antialias=False)

    def forward(self, x):
        # resize image
        x = self.resize(x)

        # layer 1
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        # layer 2
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        # layer 3
        x = F.leaky_relu(self.conv3(x))

        # flatten
        x = x.view(-1, 16*16*16)
        
        # fully connected layer
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

class IsingDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.img_labels = labels
        self.img_data = data
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        label = self.img_labels[idx]
        if self.transform=="train":
            #image = transforms.functional.rotate(image, 90)/4 + transforms.functional.rotate(image, 180)/4 + transforms.functional.rotate(image, 270)/4 + image/4
            hflipper = transforms.RandomHorizontalFlip(p=0.5)
            vflipper = transforms.RandomVerticalFlip(p=0.5)
            image = transforms.functional.rotate(image, np.random.randint(4)*90)
            image = hflipper(image)
            image = vflipper(image)
            image = torch.roll(image, shifts=(np.random.randint(64),np.random.randint(64)), dims=(-1, -2))
        if self.transform=="test":
            None
            #image = transforms.functional.rotate(image, 90)/4 + transforms.functional.rotate(image, 180)/4 + transforms.functional.rotate(image, 270)/4 + image/4
        return image, label

def test_accuracy(net, testloader, selection, test_batch_size, output_label=None, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        if np.any(output_label) == None: # if diff_array is not given, then return accuracy and do not store diff_array
            for i, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), (labels).to(device)
                outputs = net(images)
                predicted = outputs
                total += labels.size(0)
                for j, (k, l) in enumerate(zip(predicted,labels)):
                    diff = abs(k-l[0]).item()
                    correct += (diff < 0.05)
                    #correct += (abs(predicted.item() - labels.item()) < 0.01)
        else: # if diff_array is given, then return accuracy and store diff_array
            for i, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), (labels).to(device)
                outputs = net(images)
                predicted = outputs
                total += labels.size(0)
                for j, (k, l) in enumerate(zip(predicted,labels)):
                    diff = abs(k-l[selection]).item()
                    correct += (diff < 10**find_exp(l[selection].item()))
                    output_label[i*test_batch_size+j, 0] = l[0].item()
                    output_label[i*test_batch_size+j, 1] = l[1].item()
                    output_label[i*test_batch_size+j, 2] = l[2].item()
                    output_label[i*test_batch_size+j, 3] = l[3].item()
                    output_label[i*test_batch_size+j, 4] = l[4].item()
                    output_label[i*test_batch_size+j, 5] = l[5].item()
                    output_label[i*test_batch_size+j, 6] = l[6].item()
                    output_label[i*test_batch_size+j, 7] = l[7].item()
                    output_label[i*test_batch_size+j, 8] = l[8].item()
                    output_label[i*test_batch_size+j, 9] = l[9].item()
                    output_label[i*test_batch_size+j, 10] = k.item()

    return correct / total, output_label

def load_data(grid_dir="./grid_data", committor_dir="./committor_data"):  
    grid_data = torch.load(grid_dir)
    committor_data = torch.load(committor_dir)

    grid_train, grid_test, committor_train, committor_test = train_test_split(grid_data, committor_data, test_size=0.2)

    trainset = IsingDataset(grid_train, committor_train, transform="train")
    testset = IsingDataset(grid_test, committor_test, transform="test")
    test_size = len(testset)
    return trainset, testset, test_size