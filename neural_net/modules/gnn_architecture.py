import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class graph_net(nn.Module):
    def __init__(self, k_edge=2, hidden_n=2):
        super(graph_net, self).__init__()

        self.k_edge = k_edge
        self.hidden_n = hidden_n
        self.global_cv = 4*int((self.k_edge*(self.k_edge+1))/2) # coded to be equal to total number of neighbours up to k_edge away
        
        self.silu = nn.SiLU() # activation function

        self.drop = nn.Dropout(0.5) # dropout function

        self.embed = nn.Sequential( # embedding features into higher dimensional representation
            nn.Linear(1, self.global_cv),
        )

        for i in range(0, k_edge): # loops over k_edge and creates k_edge gcn layers
            self.add_module("gcn_%d" % i, GCNConv(self.global_cv, self.global_cv))

        for i in range(0, hidden_n): # loops over hidden_n and creates hidden_n fully connected layers
            self.add_module("fc_%d" % i, nn.Linear(self.global_cv, self.global_cv))

        self.output = nn.Sequential( # output layer (alpha and beta of beta distribution)
            nn.Linear(self.global_cv, 2),
        )

    def forward(self, features, edge_index, batch_size):
        x = self.embed(features)
        residual = x

        # k edge embedding
        for i in range(0, self.k_edge):
            x = self.silu(x)
            x = self.drop(x)
            x = self._modules["gcn_%d" % i](x, edge_index)
            x = x + residual
            residual = x

        x = x.view(batch_size, -1, self.global_cv)
        x = torch.mean(x, dim=1)
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
