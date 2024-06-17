"""
============================================================================================
                                 gnn_weight_init.py

Python file containing graph neural network weight initialization function. Makes use
of Kaiming or He initialization.
 ===========================================================================================
// H. Naguszewski. University of Warwick
"""

import torch.nn as nn
from torch_geometric.nn import GCNConv

def weight_init(m):
    """weight_init
    Weight initialization for graph neural network. Function is applied to neural network 
    object like so "net.apply(weight_init)" where net is the neural network object.

    Parameters:
    m: neural network layer

    Returns:
    None
    """
    if isinstance(m, (nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    if isinstance(m, (GCNConv)):
        nn.init.kaiming_uniform_(m.lin.weight)
        m.bias.data.fill_(0.0)
