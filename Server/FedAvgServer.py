from Server.ABCServer import ABCServer

class FedAvgServer(ABCServer):

    def aggregate(clients):
        n_clients = len(clients)
        server_weights = self.get_weights()
        client_weights = [cl.get_weights() for cl in clients]
        for key in server_state:
            client_weights = [state[key] for state in client_weights]
            server_weights[key] = torch.stack(client_tens, dim=0).sum(dim=0)/n_clients
        self.set_weights(server_weights)
