from Servers.ProbabilisticServer import ProbabilisticServer
import torch
from torch.distributions import Normal

class FedAgServer(ProbabilisticServer):

    def __init__(self,model,lr=1,tau=0.1,b1=.9,b2=0.99,momentum=1,optimizer='none',meanshift=None):
        super().__init__(model,optimizer = optimizer,lr=lr,tau=tau,b1=b1,b2=b2,momentum=momentum,meanshift=None)
        w_flat = self._model_weight_to_array(self.model.get_weights())
        self.distribution = Normal(w_flat,torch.ones(len(w_flat)))


    def combine(self, client_weights,device=None, client_scaling = None):
        # Dynamically calculate mean and variance of server weights.
        device = device if device!=None else 'cpu'
        mu_n = torch.zeros(self.model_size).to(device)
        s_n = torch.zeros(self.model_size).to(device)
        M_n = torch.zeros(self.model_size).to(device)
        for n,client_w in enumerate(client_weights):
            mu_n2 = (n*mu_n+client_w)/(n+1)
            M_n= M_n+(client_w-mu_n)*(client_w-mu_n2)
            if(n!=0):
                s_n= torch.sqrt(M_n/n)
            mu_n = mu_n2

        # Add small variance to avoid zero variance
        s_n = s_n+0.0000001
        self.distribution = Normal(mu_n,s_n)

        return mu_n

    def sample_model(self):
        w_r = self.distribution.sample()
        w_r = torch.split(mu_n,self.tens_lengths)
        w_r = {k:torch.reshape(w_r[i],self.model_shapes[i]) for i,k in enumerate(w)}
        return w_r
