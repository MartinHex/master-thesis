from Clients.Base_Client import Base_Client
from torch import optim
import torch
from collections import defaultdict

class FedPaClient(Base_Client):
    def __init__(self, model, dataloader, learning_rate = 0.01, burn_in =  0,
            shrinkage = 1,momentum=0,decay=0,dampening=0):
        super(FedPaClient, self).__init__(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate,
                                momentum=momentum,weight_decay=decay,dampening=dampening)
        self.dataloader = dataloader

        # Initiate model architecture for flattening
        w = model.get_weights()
        self.layers = list(w)
        # Set tensorlengths for future reconstruction of flattening.
        self.layer_size = [len(w[k].flatten()) for k in w]
        self.model_size = sum(self.layer_size)
        self.layer_shapes = [w[k].size() for k in w]

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.dampening = dampening

        # FedPa parameters
        self.beta = shrinkage
        self.shrinkage = shrinkage

    def train(self,epochs=1, device = None):
        # Produce the desired gradient online and at any-time as described in appendix C by Al-Shedivat et al.
        # (https://arxiv.org/pdf/2010.05273.pdf) mostly copied from:
        # https://github.com/alshedivat/fedpa/blob/master/federated/inference/local.py#L170-L226
        initial_weights = self.model.get_weights()
        initial_weights = self._model_weight_to_array(initial_weights).to(device)
        num_samples = epochs
        samples = []
        for epoch in range(epochs):
            samples.append(self._produce_iasg_sample(device))
        rho = self.shrinkage
        dp = defaultdict(list)
        samples_ra = samples[0]
        delta = initial_weights - samples_ra
        for t, s in enumerate(samples[1:], 2):
            u = v = s - samples_ra
            # Compute v_{t-1,t} (solution of `sigma_{t-1} x = u_t`).
            for k, (v_k, dot_uk_vk) in enumerate(zip(dp["v"], dp["dot_u_v"]), 2):
                gamma_k = rho * (k - 1) / k
                v = v - gamma_k * torch.dot(v_k, u) / (1 + gamma_k * dot_uk_vk) * v_k
            # Compute `dot(u_t, v_t)` and `dot(u_t, delta_t)`.
            dot_u_v = torch.dot(u, v)
            dot_u_d = torch.dot(u, delta)
            # Compute delta.
            gamma = rho * (t - 1) / t
            c = gamma * (t * dot_u_d - dot_u_v) / (1 + gamma * dot_u_v)
            delta -= (1 + c) * v / t
            # Update the DP state.
            dp["v"].append(v)
            dp["dot_u_v"].append(dot_u_v)
            # Update running mean of the samples.
            samples_ra = ((t - 1) * samples_ra + s) / t
        final_delta = delta * (1 + (num_samples - 1) * rho)
        new_weights = initial_weights.add(final_delta)
        new_weights = self._array_to_model_weight(new_weights)
        return new_weights

    def _produce_iasg_sample(self, device):
        sample_weight = dict()
        for i, weight in enumerate(self.model.iter_train_model(self.dataloader,self.optimizer, device = device)):
            for key in weight:
                if (i == 0):
                    sample_weight[key] = weight[key]
                else:
                    sample_weight[key] = weight[key].add(sample_weight[key], alpha = i).div((i + 1))
        return self._model_weight_to_array(sample_weight).to(device)


    def _model_weight_to_array(self,w):
        return torch.cat([w[k].flatten() for k in w]).detach()

    def _array_to_model_weight(self,a):
        a_tens = torch.split(a,self.layer_size)
        model_w = {k:torch.reshape(a_tens[i],self.layer_shapes[i])
                    for i,k in enumerate(self.layers )}
        return model_w

    def reset_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate,
            momentum = self.momentum, weight_decay = self.decay, dampening = self.dampening)
