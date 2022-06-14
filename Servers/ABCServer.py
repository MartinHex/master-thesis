from abc import ABC,abstractmethod
from Servers.optimizers.adam import Adam
from Servers.optimizers.sgd import SGD
from Servers.optimizers.yogi import Yogi
from torch import nn
import torch
import warnings

class ABCServer(ABC):

    def __init__(self,model,optimizer='none',lr=1,tau=0.1,b1=.9,b2=0.99,momentum=0,meanshift=None):
        self.model = model
        self.meanshift = meanshift
        if optimizer == 'sgd':
            self.optimizer= SGD(lr=lr,momentum=momentum,tau=tau)
        elif optimizer == 'adam':
            self.optimizer= Adam(lr=lr,tau=tau,b1=b1,b2=b2)
        elif optimizer == 'yogi':
            self.optimizer= Yogi(lr=lr,tau=tau,b1=b1,b2=b2)
        else:
            self.optimizer= None

        # Get model shapes for for future reconstruction of flattening.
        w = model.get_weights()
        self.layers = list(w)
        self.layer_size = [len(w[k].flatten()) for k in w]
        self.model_size = sum(self.layer_size)
        self.layer_shapes = [w[k].size() for k in w]

    @abstractmethod
    def combine(self,clients):
        pass

    def aggregate(self, clients,device=None, client_scaling = None):
        with torch.no_grad():
            # Extract old weights
            w_old = self._model_weight_to_array(self.get_weights()).to(device)

            # Get client  weights into single vectors
            client_weights = [self._model_weight_to_array(c.get_weights()) for c in clients]
            client_weights = torch.stack(client_weights).to(device)

            # Calculate bandwidth
            if self.meanshift in ['client-shift','mean-shift']:
                self.bandwidths = self._silverman(client_weights)

            # Do aggregation with or without mean-shifts
            if self.meanshift =='client-shift':
                client_weights = self._client_shift(client_weights,device=device)
            w_new = self.combine(client_weights,device=device, client_scaling = client_scaling)
            if self.meanshift =='mean-shift':
                w_new = self._mean_shift(client_weights,w_new,device=device)

            # Last step optimization
            if self.optimizer: w_new = self.fedOpt(w_new,w_old)

            # Clean up and set new weights
            client_weights.to('cpu')
            w_old.to('cpu')
            w_new = self._array_to_model_weight(w_new.to('cpu'))
            self.set_weights(w_new)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def push_weights(self, clients):
        for client in clients:
            client.set_weights(self.get_weights())

    def evaluate(self, dataloader, loss_func=nn.CrossEntropyLoss(), device = None, take_mean = True):
        return self.model.evaluate(dataloader, loss_func, device = device, take_mean = take_mean)

    def fedOpt(self,w_new,w_old):
        return self.optimizer.opt(w_new,w_old)

    def _model_weight_to_array(self,w):
        return torch.cat([w[k].flatten() for k in w]).detach()

    def _array_to_model_weight(self,a):
        a_tens = torch.split(a,self.layer_size)
        model_w = {k:torch.reshape(a_tens[i],self.layer_shapes[i])
                    for i,k in enumerate(self.layers )}
        return model_w

    def _client_shift(self,client_weights,device):
        res = torch.zeros(client_weights.shape)
        for i in range(len(client_weights)):
            res[i,:] = self._mean_shift(client_weights,client_weights[i],device=device)
        print(res)
        return res.to(device)

    def _mean_shift(self,client_weights,init,tol=1e-6,device=None,max_iter=100):
        w_res = init.to(device)
        non_fixed_idx = torch.std(client_weights,0).nonzero().flatten()
        for i in range(max_iter):
            # Initiate parameters which are to be mean-shifted
            w = w_res[non_fixed_idx]
            n_nonzeros = len(w)
            if n_nonzeros==0: break
            H = self.bandwidths[non_fixed_idx]
            denominator= torch.zeros(n_nonzeros).to(device)
            numerator = torch.zeros(n_nonzeros).to(device)
            for _,client_w in enumerate(client_weights):
                w_i = client_w[non_fixed_idx].to(device)
                K = self._epanachnikov_kernel((w-w_i)/H)
                denominator += K
                numerator += K*w_i
            # Calculate the mean shift
            m_x = numerator/denominator
            # Replace nan values with previus iteration
            nan_idx = m_x.isnan().nonzero().flatten()
            m_x[nan_idx] = w[nan_idx]
            # Update resulting parameters
            w_res[non_fixed_idx] = m_x
            # Update which parameters which are to be selected for next iteration
            non_converged_idx = torch.abs(w-m_x)>tol
            non_fixed_idx = non_fixed_idx[non_converged_idx]
        if(i==max_iter-1):
            warnings.warn("Maximal iteration reacher. You may want to look into increasing the amount of iterations.")
        return w_res

    def _gaussian_kernel(self,u):
        return torch.exp(-u**2/2)

    def _epanachnikov_kernel(self,u):
        u[torch.abs(u)>1] = 1
        return 3/4 * (1 - u**2)

    def _scott(self,client_weights):
        n = len(client_weights)
        h = torch.std(client_weights,0)
        return n**(-0.2)*h

    def _silverman(self,client_weights):
        n = len(client_weights)
        h = torch.std(client_weights,0)
        iqr = torch.quantile(client_weights,0.75,dim=0)-torch.quantile(client_weights,0.25,dim=0)
        h[h>iqr] = iqr[h>iqr]
        return (n*3/4)**(-0.2)*h
