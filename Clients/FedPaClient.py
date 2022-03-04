from Clients.Base_Client import Base_Client
from torch import optim
import torch
from collections import defaultdict

class FedPaClient(Base_Client):
    def __init__(self, model, dataloader, learning_rate = 0.01, burn_in =  0,
                K = 1, shrinkage = 1, mcmc_samples = 1,,momentum=0,decay=0,dampening=0):
        super(FedPaClient, self).__init__(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate,
                                momentum=momentum,decay=decay,dampening=dampening)
        self.dataloader = dataloader
        self.burn_in = burn_in
        self.K = K
        self.mcmc_samples = mcmc_samples
        self.beta = shrinkage
        self.shrinkage = shrinkage

    def train(self):
        initial_weights = self.model.get_weights()
        # Burn in phase
        self.model.train_model(self.dataloader,self.optimizer,epochs = self.burn_in) if self.burn_in > 0 else None
        weights_after_burn_in = self.model.get_weights()
        current_delta = dict()
        current_average = dict()
        u = defaultdict(lambda: [])
        v = defaultdict(lambda: [])
        # Produce the desired gradient online and at any-time as described in appendix C by Al-Shedivat et al.
        # (https://arxiv.org/pdf/2010.05273.pdf)
        for i in range(self.mcmc_samples):
            t = i + 1
            weight_sample = self._produce_iasg_sample()
            for key in weight_sample.keys():
                if i == 0:
                    current_average[key] = weight_sample[key]
                    current_delta[key] = weights_after_burn_in[key].sub(current_average[key])
                    v[key].append(torch.empty(weight_sample[key].shape))
                    u[key].append(torch.empty(weight_sample[key].shape))
                else:
                    u[key].append(weight_sample[key].sub(current_average[key]))
                    for k in range(i - 1):
                        gamma_k = (self.beta * k) / (k + 1) #Index different due to python start @ 0, Equation 17.
                        nominator = (gamma_k * torch.flatten(v[key][k]).matmul(torch.flatten(u[key][-1])).item())
                        denominator = (1 + gamma_k * torch.flatten(v[key][k]).matmul(torch.flatten(u[key][k])).item())
                        if k == 1:
                            sum = v[key][k].mul(nominator / denominator)
                        else:
                            sum = sum.add(v[key][k].mul(nominator / denominator))
                    v[key].append(u[key][i].sub(sum)) #Equation 28
                    current_average[key] = weight_sample[key].add(current_average, alpha = i).div(i + 1)
                    gamma_t = (self.beta * (t - 1)) / t #Equation 17.
                    nominator = gamma_t * (t * torch.flatten(u[key][i]).matmul(current_delta) - torch.flatten(u[key][i]).matmul(v[key][i]))
                    denominator = 1 + gamma_t * torch.flatten(u[key][i]).matmul(torch.flatten(v[key][i]))
                    current_delta[key] = current_delta[key].sub((1 + nominator / denominator).mul(v[key][i]).div(i + 1)) #Equation 23
                    current_average = weight_sample[key].add(current_average[key], alpha = i).div(i + 1)
        new_weights = dict()
        for key in initial_weights.keys():
            new_weights[key] = initial_weights[key].sub(current_delta[key] / self.shrinkage) #Gradient from Equation 24
        return new_weights

    def _produce_iasg_sample(self):
        averaged_weights = dict()
        for i in range(self.K):
            self.model.train_model(self.dataloader,self.optimizer,epochs = 1)
            weights = self.model.get_weights()
            for key in weights.keys():
                if i == 0:
                    averaged_weights[key] = weights[key]
                else:
                    averaged_weights[key] = weights[key].add(averaged_weights[key], alpha = i).div(i + 1)
        return averaged_weights
