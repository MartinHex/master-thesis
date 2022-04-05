from Servers.ABCServer import ABCServer
from collections import defaultdict
import torch

class FedAvgServer(ABCServer):

    def combine(self, clients, device = None, client_scaling = None):

        # Calculate scaling
        if client_scaling:
            normalizing = sum(client_scaling)
        else:
            normalizing = len(clients)
            client_scaling = torch.ones(normalizing)

        client_weights = [cl.get_weights() for cl in clients]
        n_clents = len(client_weights)
        new_weights = defaultdict(lambda:0)
        for i,cw in enumerate(client_weights):
            for k in cw:
                new_weights[k]+=cw[k]/n_clents

        return new_weights
