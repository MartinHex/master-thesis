from Clients.Base_Client import Base_Client
from torch import optim
import torch
from collections import defaultdict

class FedPaClient(Base_Client):
    def __init__(self, model, dataloader, learning_rate = 0.01, burn_in =  0,
                K = 1, shrinkage = 1, mcmc_samples = 1,momentum=0,decay=0,dampening=0):
        super(FedPaClient, self).__init__(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate,
                                momentum=momentum,weight_decay=decay,dampening=dampening)
        self.dataloader = dataloader
        self.burn_in = burn_in
        self.K = K
        self.mcmc_samples = mcmc_samples
        self.beta = shrinkage
        self.shrinkage = shrinkage

    def train(self,epochs=1, device = None):
        # Produce the desired gradient online and at any-time as described in appendix C by Al-Shedivat et al.
        # (https://arxiv.org/pdf/2010.05273.pdf)
        initial_weights = self.model.get_weights()
        burn_in_steps = int(epochs * len(self.dataloader) * self.burn_in)
        weights_after_burn_in = dict()
        current_delta = dict()
        current_average = dict()
        v = defaultdict(lambda: [])
        denominators = defaultdict(lambda: [])
        sample_done = False
        t = 0
        sample_weight = dict()
        for i, weights in enumerate(self.model.iter_train_model(self.dataloader,self.optimizer,epochs = epochs)):
            # Check if last step of burn in
            if (i == burn_in_steps):
                for key in weights.keys():
                    weights_after_burn_in[key] = weights[key].to(device)

            # Check if we are producing IASG sample
            is_not_burn_in = (i >= burn_in_steps)
            if (is_not_burn_in and (not sample_done)):
                tmp = (i - burn_in_steps + 1)%self.K
                sample_done = self._produce_iasg_sample(tmp, weights, sample_weight, device)

            # If new IASG sample we update moving delta.
            l = (i - burn_in_steps) // self.K
            if is_not_burn_in and sample_done:
                t += 1
                for key in sample_weight.keys():
                    if l == 0:
                        current_average[key] = initial_weights[key].to(device)
                        current_delta[key] = weights_after_burn_in[key].sub(current_average[key])
                        v[key].append(torch.empty(sample_weight[key].shape).to(device))
                    else:
                        u_key = sample_weight[key].sub(current_average[key])
                        sum = torch.zeros(sample_weight[key].shape).to(device)
                        for k in range(l - 1):
                            gamma_k = (self.beta * k) / (k + 1) #Index different due to python start @ 0, Equation 17.
                            nominator = (gamma_k * torch.flatten(v[key][k]).matmul(torch.flatten(u_key)).item())
                            if k == l - 2:
                                denominators[key].append(1 + gamma_k * torch.flatten(v[key][k]).matmul(torch.flatten(u_key)).item())
                            denominator = denominators[key][k]
                            sum = sum.add(v[key][k].mul(nominator / denominator))
                        v[key].append(u_key.sub(sum)) #Equation 28
                        current_average[key] = sample_weight[key].add(current_average[key], alpha = l).div(l + 1)
                        gamma_t = (self.beta * (t - 1)) / t #Equation 17.
                        nominator = gamma_t * (t * torch.flatten(u_key).matmul(torch.flatten(current_delta[key])) - torch.flatten(u_key).matmul(torch.flatten(v[key][l])))
                        denominator = 1 + gamma_t * torch.flatten(u_key).matmul(torch.flatten(v[key][l]))
                        current_delta[key] = current_delta[key].sub((1 + nominator / denominator).mul(v[key][l]).div(l + 1)) #Equation 23
                        current_average[key] = sample_weight[key].add(current_average[key], alpha = l).div(l + 1)
                sample_done = False
        new_weights = dict()
        for key in initial_weights.keys():
            new_weights[key] = initial_weights[key].sub(current_delta[key].cpu() / self.shrinkage) #Gradient from Equation 24
        return new_weights

    def _produce_iasg_sample(self, tmp, weight, averaged_weights, device):
        for key in weight.keys():
            if ((tmp == 1) or (self.K == 1)):
                averaged_weights[key] = weight[key].to(device)
            else:
                averaged_weights[key] = weight[key].to(device).add(averaged_weights[key], alpha = (tmp - 1)).div((tmp - 1) + 1)
        return ((self.K == 1) or (tmp == 0)) #Return whether the 'K-loop' is done or not.
