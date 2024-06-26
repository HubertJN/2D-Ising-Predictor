"""
============================================================================================
                                 cnn_architecture.py

Python file containing convolutional neural network architecture written using PyTorch.
File includes initialization function for creating the cnn as well as a forward pass
function.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class net_init(nn.Module):
    def __init__(self, conv, hidden_n, img_dim=64):
        """__init__
        Convolutional neural network initialization.

        Parameters:
        conv: number of convolutions to be performed
        hidden_n: number of fully connected linear layers after convolution layer(s)
        img_dim: size of x and y dimension for image rescaling (optional, default is 64)

        Returns:
        None
        """
        super(net_init, self).__init__()

        self.conv = conv
        self.hidden_n = hidden_n
        self.img_dim = img_dim
        self.channels = int((self.conv*(self.conv+1))/2) # coded to be equal to total number of neighbours 
        
        self.resize = transforms.Resize((self.img_dim, self.img_dim), antialias=False) # resize input tensor

        self.silu = nn.SiLU() # activation function

        self.drop = nn.Dropout(0.25) # dropout function

        self.pool = nn.MaxPool2d(2,2) # pooling function

        self.add_module("cn_0", nn.Conv2d(1, self.channels, 3, padding=[1,1], padding_mode="circular"))

        for i in range(1, conv): # loops over conv and creates conv cn layers
            self.add_module("cn_%d" % i, nn.Conv2d(self.channels, self.channels, 3, padding=[1,1], padding_mode="circular"))

        conv_output_nodes = int((self.img_dim/(2**conv))**2)*self.channels    # input is pooled conv times, each time halving
                                                                # the size of input tensor in both dimensions
        
        for i in range(0, hidden_n): # loops over hidden_n and creates hidden_n fully connected layers
            self.add_module("fc_%d" % i, nn.Linear(conv_output_nodes, conv_output_nodes))

        self.output = nn.Sequential( # output layer (alpha and beta of beta distribution)
            nn.Linear(conv_output_nodes, 2),
        )

        print("channels: %d" % self.channels)
        print("conv_output_nodes: %d" % conv_output_nodes)

    def forward(self, image):
        """forward
        Forward pass of convolutional neural network.

        Parameters:
        image: input image

        Returns:
        (alpha, beta): alpha and beta parameters of beta distribution in 1D array
        """
        x = self.resize(image) # resize input image to img_dim x img_dim
        residual = x

        # convolutions
        for i in range(0, self.conv):
            x = self.silu(x)
            x = self.drop(x)
            x = self._modules["cn_%d" % i](x)
            x = x + residual
            x = self.pool(x)
            residual = x

        x = torch.flatten(x, start_dim = 1) # flatten
        residual = x

        # fully connected layers
        for i in range (0, self.hidden_n):
            x = self.silu(x)
            x = self.drop(x)
            x = self._modules["fc_%d" % i](x)
            x = x + residual
            residual = x
        
        x = self.silu(x)
        x = self.drop(x)
        x = self.output(x)

        return abs(x.squeeze(1)) + 0.0001 # mapped to absolute value due to alpha,beta > 0

