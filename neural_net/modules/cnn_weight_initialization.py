import torch.nn as nn

def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.0)