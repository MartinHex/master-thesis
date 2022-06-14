from Servers.ABCServer import ABCServer
from collections import defaultdict
import torch

class FedAvgServer(ABCServer):

    def combine(self, client_weights, device = None, client_scaling = None):

        # Calculate scaling
        if not client_scaling:
            client_scaling = torch.ones(len(client_weights))
        weights = client_scaling/sum(client_scaling)

        # Calculate new weight
        new_weights = torch.zeros(self.model_size).to(device)
        for client_w,w in zip(client_weights,weights):
            new_weights += w*client_w

        return new_weights
