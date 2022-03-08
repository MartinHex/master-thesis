from Servers.ProbabilisticServer import ProbabilisticServer
import torch
from torch.distributions import Normal

class FedAgServer(ProbabilisticServer):

    def __init__(self,model):
        super().__init__(model)
        w = model.get_weights()
        # Set initial distributions
        w_flat = torch.cat([w[k].flatten() for k in w])
        self.distribution = Normal(w_flat,torch.ones(len(w_flat)))
        # Set tensorlengths for future reconstruction of flattening.
        self.tens_lengths = [len(w[k].flatten()) for k in w]
        self.model_shapes = [w[k].size() for k in w]
        self.model_size = sum(self.tens_lengths)

    def aggregate(self, clients,device=None):
        # Dynamically calculate mean and variance of server weights.
        device = device if device!=None else 'cpu'
        mu_n = torch.zeros(self.model_size).to(device)
        s_n = torch.zeros(self.model_size).to(device)
        M_n = torch.zeros(self.model_size).to(device)
        for n,client in enumerate(clients):
            w = client.get_weights()
            w_flat = torch.cat([w[k].flatten() for k in w]).to(device)
            mu_n2 = (n*mu_n+w_flat)/(n+1)
            M_n= M_n+(w_flat-mu_n)*(w_flat-mu_n2)
            if(n!=0):
                s_n= torch.sqrt(M_n/n)
            mu_n = mu_n2

        # Add small variance to avoid zero variance
        s_n = s_n+0.0000001
        self.distribution = Normal(mu_n,s_n)

        # reconstruct weights using EM esstimate (mean)
        w_r = torch.split(mu_n,self.tens_lengths)
        w_r = {k:torch.reshape(w_r[i],self.model_shapes[i]) for i,k in enumerate(w)}
        self.set_weights(w_r)
        self.model.to('cpu')

    def sample_model(self):
        w_r = self.distribution.sample()
        w_r = torch.split(mu_n,self.tens_lengths)
        w_r = {k:torch.reshape(w_r[i],self.model_shapes[i]) for i,k in enumerate(w)}
        return w_r
