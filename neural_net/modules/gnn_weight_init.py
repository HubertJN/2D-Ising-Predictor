import torch.nn as nn
from torch_geometric.nn import GCNConv

def weight_init(m):
    if isinstance(m, (nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    if isinstance(m, (GCNConv)):
        nn.init.kaiming_uniform_(m.lin.weight)
        m.bias.data.fill_(0.0)
