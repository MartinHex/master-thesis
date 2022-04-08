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
        # (https://arxiv.org/pdf/2010.05273.pdf) mostly translated from:
        # https://github.com/google-research/federated/tree/master/posterior_averaging
        x_0 = self.model.get_weights()
        num_samples = epochs
        samples = []
        for epoch in range(epochs):
            samples.append(self._produce_iasg_sample(device))
        with torch.no_grad():
            rho = self.shrinkage

            # First sample
            weights_sample_mean = samples[0]
            weights_delta_tilde = dict()
            for key in weights_sample_mean:
                weights_delta_tilde[key] = x_0[key].to(device) - weights_sample_mean[key]

            # Second Sample
            un_flat = dict()
            vn_flat = dict()
            for key in samples[1]:
                un_flat[key] = (samples[1][key] - weights_sample_mean[key]).flatten()
                vn_flat[key] = (samples[1][key] - weights_sample_mean[key]).flatten()

            # Compute `dot(vn, un)`.
            dot_vn_un = 0
            for key in un_flat:
                dot_vn_un += torch.dot(vn_flat[key], un_flat[key])

            # Initiate the history of vk_tas & dot_vk_uk_ta (recursion state in tff implementation)
            vk_tas = defaultdict(list)
            for key in vn_flat:
                vk_tas[key].append(vn_flat[key])
            dot_vk_uk_ta = [dot_vn_un]

            # Iterate through all other samples, excluding samples 1 and 2
            for n, sample in enumerate(samples[2:], 3):
                # Update the running mean of the weights samples.
                for key in weights_sample_mean:
                    weights_sample_mean[key] = ((n - 1) * weights_sample_mean[key] + sample[key]) / n


                # Compute u_{n} (deviation of the new sample from the previous mean).
                un_flat = dict()
                for key in sample:
                    un_flat[key] = (sample[key] - weights_sample_mean[key]).flatten()

                # Compute v_{n-1, n} (solution of `sigma_{n-1} x = u_n`).
                # Step 1: compute `vk_coeff = gamma * dot(v_k, u_n) / (1 + gamma * uv_k)`.
                gammas_range = 2 + torch.arange(0, n - 2).to(device)
                gammas = rho * (gammas_range - 1) / gammas_range
                dot_vk_un = dict()
                for key in un_flat:
                    dot_vk_un[key] = torch.einsum('ij,j->i', torch.stack(vk_tas[key]), un_flat[key])
                dot_vk_un_tmp = 0
                for key in dot_vk_un:
                    dot_vk_un_tmp += torch.sum(dot_vk_un[key].flatten())
                dot_vk_un = dot_vk_un_tmp
                dot_vk_uk = torch.stack(dot_vk_uk_ta)
                vk_coeffs = gammas * dot_vk_un / (1 + gammas * dot_vk_uk)

                # Step 2: compute `vn = u - sum_k vk_coeff * vk` and `dot(v_n, u_n)`.
                vn_flat = dict()
                for key in un_flat:
                    vn_flat[key] = un_flat[key] - torch.einsum('i,ij->j', vk_coeffs, torch.stack(vk_tas[key]))

                # Compute `dot(vn, un)`.
                dot_vn_un = 0
                for key in sample:
                    dot_vn_un += torch.dot(vn_flat[key], un_flat[key])

                # Update the history of vk_tas & dot_vk_uk_ta (tff: recursion state)
                for key in vk_tas:
                    vk_tas[key].append(vn_flat[key])
                dot_vk_uk_ta.append(dot_vn_un)

                # Compute weights delta tilde: `weights_delta_tilde += coeff * vn / n`.
                weight_delta_tilde_flat = dict()
                dot_wd_un = 0
                for key in weights_delta_tilde:
                    weight_delta_tilde_flat[key] = weights_delta_tilde[key].flatten()
                    dot_wd_un += torch.dot(weight_delta_tilde_flat[key], un_flat[key])
                gamma = rho * (n - 1) / n
                vn_coeff = 1. - gamma * (n * dot_wd_un + dot_vn_un) / (1. + gamma * dot_vn_un)
                for key in weights_delta_tilde:
                    weights_delta_tilde[key] = weights_delta_tilde[key] + vn_coeff * torch.reshape(vn_flat[key], weights_delta_tilde[key].shape) / n

            # Obtain new weights delta by rescaling weights delta tilde.
            weights_delta = dict()
            for key in weights_delta_tilde:
                weights_delta[key] = weights_delta_tilde[key] * (1 + (n - 1) * rho)
                norm += torch.sum(torch.square(weights_delta[key]))

            # Update from gradient to model weights.
            new_weights = dict()
            for key in weights_delta:
                new_weights[key] = x_0[key].sub(weights_delta[key].cpu())
            self.set_weights(new_weights)

    def _produce_iasg_sample(self, device):
        sample_weight = dict()
        for i, weight in enumerate(self.model.iter_train_model(self.dataloader,self.optimizer, device = device)):
            for key in weight:
                if (i == 0):
                    sample_weight[key] = weight[key]
                else:
                    sample_weight[key] = weight[key].add(sample_weight[key], alpha = i).div((i + 1))
        return sample_weight

    def reset_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate,
            momentum = self.momentum, weight_decay = self.decay, dampening = self.dampening)
