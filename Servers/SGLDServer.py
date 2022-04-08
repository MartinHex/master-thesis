from Servers.ABCServer import ABCServer
import numpy as np
from torch import nn
import torch

class SGLDServer(ABCServer):
    def __init__(self,training_model, evaluation_model, burn_in, optimizer='none', lr=1, tau=0.1, b1=.9, b2=0.99, momentum=1, client_lr = None):
        self.client_lr = client_lr
        self.training_model = training_model
        self.burn_in = burn_in
        self.total_rounds = 0
        self.burn_in_complete = False
        super().__init__(evaluation_model, optimizer = optimizer, lr = lr, tau = tau, b1 = b1, b2 = b2, momentum = momentum)

    def combine(self, clients, device = None, client_scaling = None):
        with torch.no_grad():
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
                # Add noise according to SGLD: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.3813&rep=rep1&type=pdf
                server_weights[key] = server_weights[key] + resultant + np.random.normal(0, 2 * self.client_lr, size = resultant.shape)

            if self.burn_in_complete: self._update_average_model()
            if self.total_rounds == self.burn_in:
                self.model.set_weights(self.get_weights())
                self.burn_in_complete = True
            elif self.total_rounds < self.burn_in:
                self.model.set_weights(self.get_weights())
            self.total_rounds += 1

            return server_weights

    def _update_average_model(self):
        current_average_weights = self.model.get_weights()
        new_weights = self.training_model.get_weights()
        iterations_past_burn_in = self.total_rounds - self.burn_in
        for key in new_weights:
            current_average_weights[key] = (current_average_weights[key] * iterations_past_burn_in).add(new_weights[key]).div(iterations_past_burn_in + 1)
        self.model.set_weights(current_average_weights)

    def get_weights(self):
        return self.training_model.get_weights()

    def set_weights(self, weights):
        self.training_model.set_weights(weights)

    def evaluate(self, dataloader, loss_func=nn.CrossEntropyLoss(), device = None, take_mean = True):
        return self.model.evaluate(dataloader, loss_func, device = device, take_mean = take_mean)

    def _calculate_grad(self,w_old, w_new, scaling):
        grad = {}
        for k in w_old:
            grad[k] = scaling * w_new[k].sub(w_old[k])
        return grad
