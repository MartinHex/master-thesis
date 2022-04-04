from Servers.ABCServer import ABCServer
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

        new_weights = {}
        for key in client_weights[0]:
            client_weights_key = [client_scaling[i]*c[key] for i,c in enumerate(client_weights)]
            new_weights[key] = torch.stack(client_weights_key, dim = 0).sum(dim = 0) / normalizing

        return new_weights
