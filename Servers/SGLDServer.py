from Servers.ABCServer import ABCServer
import numpy as np
from torch import nn
import torch

class SGLDServer(ABCServer):
    def __init__(self,training_model, evaluation_model, burn_in, optimizer='none', lr=1, tau=0.1, b1=.9, b2=0.99, momentum=1, client_lr = None,meanshift=None):
        self.client_lr = client_lr
        self.training_model = training_model
        self.burn_in = burn_in
        self.total_rounds = 0
        self.burn_in_complete = False
        super().__init__(evaluation_model, optimizer = optimizer, lr = lr, tau = tau, b1 = b1, b2 = b2, momentum = momentum,meanshift=None)

    def combine(self, client_weights, device = None, client_scaling = None):
        # Calculate scaling
        if not client_scaling:
            client_scaling = torch.ones(len(client_weights))
        weights = client_scaling/sum(client_scaling)

        # Calculate new weights
        new_weights = torch.zeros(self.model_size).to(device)
        for client_w,w in zip(client_weights,weights):
            new_weights += w*client_w

        # Add random noice
        new_weights += np.random.normal(0, 2 * self.client_lr, size = self.model_size)

        if self.burn_in_complete:
            self._update_average_model(new_weights)
        else:
            self.model.set_weights(self._array_to_model_weight(new_weights))
            self.burn_in_complete = self.total_rounds >= self.burn_in
        self.total_rounds += 1

        return new_weights

    def _update_average_model(self,new_weights):
        current_average_weights = self.model.get_weights()
        new_weight = self._array_to_model_weight(new_weights)
        iterations_past_burn_in = self.total_rounds - self.burn_in
        for key in new_weights:
            current_average_weights[key] = (current_average_weights[key] * iterations_past_burn_in).add(new_weight[key]).div(iterations_past_burn_in + 1)
        self.model.set_weights(current_average_weights)

    def get_weights(self):
        return self.training_model.get_weights()

    def set_weights(self, weights):
        self.training_model.set_weights(weights)

    def evaluate(self, dataloader, loss_func=nn.CrossEntropyLoss(), device = None, take_mean = True):
        return self.model.evaluate(dataloader, loss_func, device = device, take_mean = take_mean)
