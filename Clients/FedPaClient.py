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
        # (https://arxiv.org/pdf/2010.05273.pdf)
        initial_weights = self.model.get_weights()
        initial_weights = self._model_weight_to_array(initial_weights).to(device)
        current_delta = torch.clone(initial_weights)
        current_average = torch.zeros(self.model_size).to(device)
        v = [torch.zeros(self.model_size).to(device)]
        sum_factors = []

        for epoch in range(epochs):
            t = epoch + 1
            sample_weight = self._produce_iasg_sample(device)

            u_t = sample_weight.sub(current_average)
            sum = torch.zeros(self.model_size).to(device)
            for k in range(t - 1):
                gamma_k = (self.beta * k) / (k + 1) #Index different due to python start @ 0, Equation 17.
                nominator = (gamma_k * v[k].matmul(u_t))
                if k == t - 2:
                    denominator = 1 + gamma_k * v[k].matmul(u_t)
                    sum_factors.append(gamma_k/denominator)
                factor = sum_factors[k]
                sum = sum.add(v[k].mul(sum_factors[k]))
            v.append(u_t.sub(sum)) #Equation 28
            current_average = sample_weight.add(current_average, alpha = t).div(t+1)
            gamma_t = (self.beta * (t - 1)) / t #Equation 17.
            nominator = gamma_t * (t * u_t.matmul(current_delta) - u_t.matmul(v[t]))
            denominator = 1 + gamma_t * u_t.matmul(v[t])
            current_delta = current_delta.sub((1 + nominator / denominator).mul(v[t]).div(t + 1)) #Equation 23
            current_average = sample_weight.add(current_average, alpha = t).div(t + 1)

        new_weights =  initial_weights.sub(current_delta / self.shrinkage).cpu() #Gradient from Equation 2
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
