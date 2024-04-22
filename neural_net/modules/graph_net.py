import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class graph_net(nn.Module):
    def __init__(self, k_edge=2, hidden_n=2, hidden_m=2):
        super(graph_net, self).__init__()

        self.k_edge = k_edge
        self.hidden_n = hidden_n
        self.hidden_m = hidden_m

        self.silu = nn.SiLU() # activation function

        self.drop = nn.Dropout(0.5)

        self.embed = nn.Linear(1, hidden_n) # embedding features into higher dimensional representation

        for i in range(0, k_edge): # loops over k and creates k gcn layers
            self.add_module("gcn_%d" % i, GCNConv(hidden_n, hidden_n))

        self.graph_dec = nn.Sequential( # graph decoding
            nn.Linear(hidden_n, hidden_m),
            self.drop,
            self.silu,
        )

        self.dec_process = nn.Sequential( # processing decoding
            nn.Linear(hidden_m, hidden_m),
            self.drop,
            self.silu,
            nn.Linear(hidden_m, hidden_m),
            self.drop,
            self.silu,
            nn.Linear(hidden_m, hidden_m),
            self.drop,
            self.silu,
        )

        self.output = nn.Sequential( # output layer (alpha and beta of beta distribution)
            nn.Linear(hidden_m, 2),
        )

    def forward(self, features, edge_index, batch_size):
        x = self.embed(features)

        # k edge embedding
        for i in range(0, self.k_edge):
            x = self.drop(self.silu(self._modules["gcn_%d" % i](x, edge_index)))

        x = x.view(batch_size, -1, self.hidden_n)
        x = torch.mean(x, dim=1)

        x = self.graph_dec(x)
        x = self.dec_process(x)
        x = self.output(x)

        return abs(x.squeeze(1)) # mapped to absolute value due to alpha,beta > 0