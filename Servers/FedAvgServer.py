from Servers.ABCServer import ABCServer
import torch

class FedAvgServer(ABCServer):

    def combine(self, clients, device = None, client_scaling = None):
        if client_scaling:
            normalizing = sum(client_scaling)
        else:
            normalizing = len(clients)
        server_weights = self.get_weights()
        client_weights = [cl.get_weights() for cl in clients]

        gradients = []
        for i, weights in enumerate(client_weights):
            if client_scaling:
                gradients.append(self._calculate_grad(server_weights, weights, client_scaling[i]))
            else:
                gradients.append(self._calculate_grad(server_weights, weights, 1))

        for key in server_weights:
            client_gradients = [gradient[key] for gradient in gradients]
            resultant = torch.stack(client_gradients, dim = 0).sum(dim = 0) / normalizing
            server_weights[key] = server_weights[key] + resultant

            #client_tensor = [state[key] for state in client_weights]
            #server_weights[key] = torch.stack(client_tensor, dim=0).sum(dim=0)/n_clients
        return server_weights

    def _calculate_grad(self,w_old, w_new, scaling):
        grad = {}
        for k in w_old:
            grad[k] = scaling * w_new[k].sub(w_old[k])
        return grad
