import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

def load_data():
    image_dir = "./training_data/image_data_base"
    label_dir = "./training_data/label_data_base"
    image_data = torch.load(image_dir)
    label_data = torch.load(label_dir)[:,0]

    # sort data
    ordering = label_data.argsort()
    label_data = label_data[ordering]
    image_data = image_data[ordering]

    # map images to [0,1]
    image_data -= image_data.min()
    image_data /= image_data.max()

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

    image_train, image_test, label_train, label_test = train_test_split(image_data[tmp_idx], label_data[tmp_idx], test_size=0.2, random_state=1)
    image_train, image_val, label_train, label_val = train_test_split(image_train, label_train, test_size=0.2, random_state=1)

    trainset = ising_dataset(image_train, label_train, train=True)
    valset = ising_dataset(image_val, label_val)
    testset = ising_dataset(image_test, label_test)

    train_size = len(trainset)
    val_size = len(valset)
    test_size = len(testset)

    return trainset, valset, testset, train_size, val_size, test_size

class ising_dataset(Dataset):
    def __init__(self, data, labels, train=False):
        self.img_labels = labels
        self.img_data = data
        self.train = train

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        label = self.img_labels[idx]
        if self.train==True:
            #image = transforms.functional.rotate(image, 90)/4 + transforms.functional.rotate(image, 180)/4 + transforms.functional.rotate(image, 270)/4 + image/4
            hflipper = transforms.RandomHorizontalFlip(p=0.5)
            vflipper = transforms.RandomVerticalFlip(p=0.5)
            image = transforms.functional.rotate(image, np.random.randint(4)*90)
            image = hflipper(image)
            image = vflipper(image)
            image = torch.roll(image, shifts=(np.random.randint(image.shape()[-1]),image.shape()[-2]), dims=(-1, -2))
        
        return image, label