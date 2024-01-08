import torch
import torch.nn as nn
import torch.nn.functional as F

class graph_net(nn.Module):
    def __init__(self, k=1, N=10, M=10):
        super(graph_net, self).__init__()

        # Cartesian coordinate transform to high dimensions
        self.cart1 = nn.Linear(3, N)
        self.cart2 = nn.Linear(N, N)

        # Neighbour list transform to high dimensions
        self.neigh1 = nn.Linear(2, N)
        self.neigh2 = nn.Linear(N, N)

        # Cartesian and Neighbour combination
        self.comb = nn.Linear(N+M, N)

        # Decoding layer
        self.deco = nn.Linear(N, 1)

        # constants
        self.k = k
        self.N = N
        self.M = M

    def forward(self, cart, neigh_list):
        cart_size = cart.size(dim=0)
        final_graph = torch.empty(cart_size, self.N)

        for i in range(cart_size):
            # Cartesian coordinate transform to high dimensions
            cart = self.cart1(cart)
            cart = self.cart2(cart)

            # Neighbour list transform to high dimensions
            neigh_size = neigh_list[0].size(dim=0)
            rij_store = torch.empty(neigh_size, self.M)
            for j in range(neigh_size):
                rij = self.neigh1(neigh_list[j])
                rij = self.neigh2(rij)
                rij_store[j] = rij

            rij_store = F.interpolate(rij_store, self.N)

            # k edge embedding
            comb_in = torch.cat((rij_store, cart), 0)
            for j in range(self.k):
                comb_out = self.comb(comb_in)
                comb_in = torch.cat((rij_store, comb_out), 0)

            final_graph[i] = comb_out

        final_graph = torch.sum(final_graph, 1)
        
        x = self.deco(final_graph)

        return x