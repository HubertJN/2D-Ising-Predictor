import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class graph_net(nn.Module):
    def __init__(self, k_edge=2, hidden_n=2, hidden_m=2):
        super(graph_net, self).__init__()

        self.k_edge = k_edge
        self.hidden_n = hidden_n
        self.hidden_m = hidden_m

        self.silu = nn.SiLU()
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(0.5)

        self.embed = nn.Linear(1, hidden_n)

        for i in range(0, k_edge):
            self.add_module("gcn_%d" % i, GCNConv(hidden_n, hidden_n))

        self.graph_dec = nn.Sequential(
            nn.Linear(hidden_n, hidden_m),
            self.drop,
            self.silu,
        )

        self.dec_process = nn.Sequential(
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

        self.output = nn.Sequential(
            nn.Linear(hidden_m, 2),
        )


    def forward(self, features, edge_index, n_nodes):
        x = self.embed(features)

        # k edge embedding
        for i in range(0, self.k_edge):
            x = self.drop(self.silu(self._modules["gcn_%d" % i](x, edge_index)))

        x = x.view(-1, n_nodes, self.hidden_n)
        x = torch.mean(x, dim=1)

        x = self.graph_dec(x)
        x = self.dec_process(x)
        x = self.output(x)

        return abs(x.squeeze(1))


#class graph_net(nn.Module):
#    def __init__(self, k_edge=2, hidden_n=2):
#        super(graph_net, self).__init__()
#
#        self.k_edge = k_edge
#        self.hidden_n = hidden_n
#
#        self.embed = nn.Linear(1, hidden_n)
#
#        self.act_fn = nn.SiLU()
#
#        for i in range(0, k_edge):
#            self.add_module("gcn_%d" % i, GCNConv(hidden_n, hidden_n))
#
#        self.node_dec = nn.Sequential(
#            nn.Linear(hidden_n, hidden_n),
#            self.act_fn,
#            nn.Linear(hidden_n, hidden_n),
#        )
#
#        self.graph_dec = nn.Sequential(
#            nn.Linear(hidden_n, hidden_n),
#            self.act_fn,
#            nn.Linear(hidden_n, 1),
#        )
#
#        self.drop = nn.Dropout(p=0.1)

#    def forward(self, features, edge_index, n_nodes):
#        # coordinate embedding
#        x = self.embed(features)
#
#        # k edge embedding
#        for i in range(0, self.k_edge):
#            x = self._modules["gcn_%d" % i](x, edge_index)# + x
#            x = self.act_fn(x)
#
#        # node decoding
#        x = self.node_dec(x)
#
#        # sum pooling
#        x = x.view(-1, n_nodes, self.hidden_n)
#        x = torch.sum(x, dim=1) 
#        
#        # fully connected layer
#        x = self.graph_dec(x)
#
#        return x.squeeze(1)
