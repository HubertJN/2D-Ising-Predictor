import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

def load_data(image_dir="./training_data/image_data", label_dir="./training_data/label_data"):  
    image_data = torch.load(image_dir)
    label_data = torch.load(label_dir)

    image_train, image_test, label_train, label_test = train_test_split(image_data, label_data, test_size=0.2)

    trainset = ising_dataset(image_train, label_train, transform="train")
    testset = ising_dataset(image_test, label_test, transform="test")
    test_size = len(testset)
    return trainset, testset, test_size

class ising_dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.img_labels = labels
        self.img_data = data
        self.transform = transform
        self.resize = transforms.Resize((64, 64), antialias=False)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        image = self.resize(image)
        label = self.img_labels[idx]
        extra = torch.tensor((0.54, 0.07))
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
        return image, extra, label