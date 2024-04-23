import torch
import torch.nn as nn
import torchvision.transforms as transforms

class conv_net(nn.Module):
    def __init__(self, conv=3, channels=2, hidden_n=2, img_dim=64):
        super(conv_net, self).__init__()

        self.conv = conv
        self.channels = channels
        self.hidden_n = hidden_n
        self.img_dim = img_dim

        self.resize = transforms.Resize((self.img_dim, self.img_dim), antialias=False) # resize input tensor

        self.silu = nn.SiLU() # activation function

        self.drop = nn.Dropout(0.5) # dropout function

        self.pool = nn.MaxPool2d(2,2) # pooling function

                
        self.flatten = torch.flatten(start_dim=1) # flatten function

        self.add_module("cn_0", nn.Conv2d(1, channels, 3, padding=[1,1], padding_mode="circular"))
        for i in range(1, conv): # loops over conv and creates conv cn layers
            self.add_module("cn_%d" % i, nn.Conv2d(channels, channels, 3, padding=[1,1], padding_mode="circular"))

        conv_output_nodes = (self.img_dim/(2**conv))**2 # input is pooled conv times, each time halving
                                                        # the size of input tensor in both dimensions

        self.conv_decode = nn.Sequential( # convolution decoding
            nn.Linear(conv_output_nodes, hidden_n),
            self.drop,
            self.silu,
        )

        self.fc = nn.Sequential( # fully connected layers
            nn.Linear(hidden_n, hidden_n),
            self.drop,
            self.silu,
            nn.Linear(hidden_n, hidden_n),
            self.drop,
            self.silu,
        )

        self.output = nn.Sequential( # output layer (alpha and beta of beta distribution)
            nn.Linear(hidden_n, 2),
        )

    def forward(self, x):
        x = self.resize(x) # resize input image to 64x64

        # convolutions
        for i in range(0, self.conv):
            x = self._modules["cn_%d" % i](x)
            x = self.pool(x)
            x = self.silu(x)
            x = self.drop(x)

        x = self.flatten(x) # flatten

        x = self.conv_decode(x)
        x = self.fc(x)
        x = self.output(x)

        return abs(x.squeeze(1)) # mapped to absolute value due to alpha,beta > 0