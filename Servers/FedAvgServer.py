from Servers.ABCServer import ABCServer
import torch

class FedAvgServer(ABCServer):

    def aggregate(self, clients, device = None):
        n_clients = len(clients)
        server_weights = self.get_weights()
        client_weights = [cl.get_weights() for cl in clients]
        for key in server_weights:
            client_tensor = [state[key] for state in client_weights]
            server_weights[key] = torch.stack(client_tensor, dim=0).sum(dim=0)/n_clients
        self.set_weights(server_weights)
