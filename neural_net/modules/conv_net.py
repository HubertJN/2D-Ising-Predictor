import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        # fully connected layers
        self.fc1 = nn.Linear(16*16*16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        # decision layers
        self.d1 = nn.Linear(3, 100)
        self.d2 = nn.Linear(100, 100)
        self.d3 = nn.Linear(100, 1)

        # pooling
        self.pool = nn.MaxPool2d(2,2)

        # convolution layers
        self.conv1 = nn.Conv2d(1, 16, 7, padding=[3,3], padding_mode="circular")
        self.conv2 = nn.Conv2d(16, 16, 5, padding=[2,2], padding_mode="circular")
        self.conv3 = nn.Conv2d(16, 16, 3, padding=[1,1], padding_mode="circular")

        # drop out
        self.drop = nn.Dropout(p=0.1)


    def forward(self, x, y):
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

        # combine and flatten
        z = torch.cat((x,y), dim=-1)
        z = z.view(-1, 3)

        # final decision layer
        z = F.leaky_relu(self.d1(z))
        z = F.leaky_relu(self.d2(z))
        z = self.d3(z)

        return z
