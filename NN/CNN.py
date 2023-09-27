import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from math import log10, floor
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap

# finding exponent of number
def find_exp(number) -> int:
    base10 = log10(abs(number))
    return floor(base10)

# selection index for labels training
selection = 7

# 0) Load data
# Defining class for loading data
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
        if self.transform=='train':
            #image = transforms.functional.rotate(image, 90)/4 + transforms.functional.rotate(image, 180)/4 + transforms.functional.rotate(image, 270)/4 + image/4
            hflipper = transforms.RandomHorizontalFlip(p=0.5)
            vflipper = transforms.RandomVerticalFlip(p=0.5)
            image = transforms.functional.rotate(image, np.random.randint(4)*90)
            image = hflipper(image)
            image = vflipper(image)
            image = torch.roll(image, shifts=(np.random.randint(64),np.random.randint(64)), dims=(-1, -2))
        if self.transform=='test':
            None
            #image = transforms.functional.rotate(image, 90)/4 + transforms.functional.rotate(image, 180)/4 + transforms.functional.rotate(image, 270)/4 + image/4
        return image, label

# Defining function for calculating accuracy
def test_accuracy(net, testloader, output_label=None, device="cpu"):
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

# Defining function for loading data and transformations
def load_data(grid_dir="./grid_data", committor_dir="./committor_data"):  
    grid_data = torch.load(grid_dir)
    committor_data = torch.load(committor_dir)

    grid_train, grid_test, committor_train, committor_test = train_test_split(grid_data, committor_data, test_size=0.2)

    trainset = IsingDataset(grid_train, committor_train, transform='train')
    testset = IsingDataset(grid_test, committor_test)
    test_size = len(testset)
    return trainset, testset, test_size

# 1) Model
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
        self.conv1 = nn.Conv2d(1, 16, 7, padding=[3,3], padding_mode="circular") # Maybe pad to 10 cause area of cluster
        self.conv2 = nn.Conv2d(16, 16, 5, padding=[2,2], padding_mode="circular")
        self.conv3 = nn.Conv2d(16, 16, 3, padding=[1,1], padding_mode="circular")

        # drop out
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
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
    
class LinNet(nn.Module):
    def __init__(self):
        super(LinNet, self).__init__()
        # fully connected layers
        self.fc1 = nn.Linear(64*64, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

        # drop out
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        # flatten
        x = x.view(-1, 64*64)
        # fully connected layer
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

# Device configuration and setting up network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

#net = ConvNet().to(device)
#net_cpu = ConvNet().to(device_cpu)  

net = ConvNet().to(device)
net_cpu = ConvNet().to(device_cpu)  

# Printing number of parameters
total_params = sum(p.numel() for p in net.parameters())
print("Parameters: ", total_params)
#exit()

# 2) Hyper-parameters, loss and optimizer, data loading
num_epochs = 100
learning_rate = 1e-3
weight_decay = 0
train_batch_size = 100
test_batch_size = 10

# Loss and optimizer
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# Loading data
trainset, testset, test_size = load_data()
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=2, batch_size=train_batch_size)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2, batch_size=test_batch_size)

# 3) Training loop

if selection == 0:
    PATH = './model_base.pth'
elif selection == 3:
    PATH = './model_lcs.pth'
elif selection == 7:
    PATH = './model_lcs_p.pth'

# Initializing counters
accuracy = 0
revert_to_checkpoint = 0
revert_limit = 0
revert_break = 0
epoch = 0
loss_offset = 0
min_loss = np.inf
lr_power = 0

# checkpoint system
checkpoint = 1

# reset system
reset = 1

# storage for loss
loss_arr = np.zeros(num_epochs)

# Loading model
load = 0
if load == 1:
    checkpoint_model = torch.load(PATH)
    net.load_state_dict(checkpoint_model['model_state_dict'])
    optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
    epoch = checkpoint_model['epoch']
    accuracy = checkpoint_model['accuracy']
    loss_arr = np.load("loss.npy")
    net.eval()
    print("Loaded NN")

# Running training loop
run = 1
if run == 1:
    print('Beginning Training Loop')
    n_total_steps = np.int32(len(trainset)/train_batch_size)
    correct = 0
    while (1):
        epoch += 1
        if epoch > num_epochs or revert_break >= 3:
            print("Last epoch reached or revertion break threshold reached.")
            break
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            # index of labels picks what to train on
            labels = labels[:,selection].to(device)
            outputs = torch.flatten(net(images))
            loss = criterion(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        revert_to_checkpoint += 1 # Incrementing checkpoint counter
        test_loss = loss.item()
        loss_arr[epoch-1] = test_loss
        if selection == 0:
            np.save("loss_base.npy", loss_arr)
        elif selection == 3:
            np.save("loss_lcs.npy", loss_arr)
        elif selection == 7:
            np.save("loss_lcs_p.npy", loss_arr)
        # Test loss for reset
        if test_loss > 0.1 and epoch == 20 and reset == 1: # Resetting optimizer if accuracy is too low
            epoch = 1
            net = ConvNet().to(device)
            print("Loss too high, resetting network and restarting from epoch 1.")
        # Saving checkpoint
        if (min_loss > test_loss):
            revert_to_checkpoint = 0
            revert_limit = 0
            revert_break = 0
            min_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                }, PATH)
        # If no improvement after x epochs, revert to previous checkpoint
        if revert_to_checkpoint == 10 and checkpoint == 1 and epoch > 20:
            revert_to_checkpoint = 0
            revert_limit += 1
            lr_power += 1
            if revert_limit > 3 and checkpoint == 1: # Reset learning rate if no improvement after 10 checkpoint revertions
                revert_limit = 0
                lr_power = 0
                learning_rate = 1.01*learning_rate
                revert_break += 1
                checkpoint_model = torch.load(PATH)
                net.load_state_dict(checkpoint_model['model_state_dict'])
                optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
                epoch = checkpoint_model['epoch']
                optimizer.param_groups[0]["lr"] = learning_rate
                print("Revertion limit reached, resetting learning rate to adjusted initial value.")
                continue
            checkpoint_model = torch.load(PATH)
            net.load_state_dict(checkpoint_model['model_state_dict'])
            optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
            epoch = checkpoint_model['epoch']
            optimizer.param_groups[0]["lr"] = learning_rate*0.5**lr_power
            print("Checkpoint loaded. Starting from epoch: ", epoch)
        lr = optimizer.param_groups[0]["lr"]
        print (f'Epoch [{epoch}/{num_epochs}], Loss: {test_loss:.8f}, Minimum Loss: {min_loss:.6f}, Learning Rate: {lr:.10f}')


test_batch_size = 1
outputs_labels = np.zeros([test_size, 11])
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=2, batch_size=test_batch_size)
net_cpu.load_state_dict(net.state_dict())
net_cpu.eval()
_, outputs_labels = test_accuracy(net_cpu, testloader, output_label=outputs_labels)

if selection == 0:
    np.save("prediction_actual_base.npy", outputs_labels)
elif selection == 3:
    np.save("prediction_actual_lcs.npy", outputs_labels)
elif selection == 7:
    np.save("prediction_actual_lcs_p.npy", outputs_labels)

# Setup class activation map
class_activation = 0
if class_activation == 1:
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net_cpu._modules.get("conv3").register_forward_hook(hook_feature)

    params = list(net_cpu.parameters())
    weight_softmax = np.squeeze(params[0].data.numpy())

    for i, (images, labels) in enumerate(testloader):
        img = images
        break

    logit = net_cpu(img)

    features_blobs = np.array(features_blobs)

    CAMs = np.sum(features_blobs[0,0], axis=0)
    CAMs = zoom(CAMs, 64/CAMs.shape[-1])

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["figure.figsize"] = (8,8)
    plt.rcParams['figure.dpi'] = 120

    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('Greens_r')(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(1.0,0.0, ncolors)
    color_array[:,0] = 1
    color_array[:,1] = 0
    color_array[:,2] = 0

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='alpha',colors=color_array)

    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)

    plt.imshow(img[0,0], cmap='Greys_r')
    plt.imshow(CAMs, alpha = 1, cmap='alpha')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()