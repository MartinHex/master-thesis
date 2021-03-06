from Servers.ABCServer import ABCServer
import torch
from torch.distributions import Normal
from torch.nn import NLLLoss
from torch.nn.utils import clip_grad_norm_
from torchcontrib.optim import SWA
from torch.optim import SGD
import random


class FedBeServer(ABCServer):

    def __init__(self,model,loc_data,M=10,swa_lr1=0.001,swa_lr2=0.0004,meanshift=None
                    swa_freq=5,swa_epochs=1,verbose=True,lr=1,tau=0.1,b1=.9,b2=0.99,momentum=1,optimizer='none'):
        super().__init__(model,optimizer = optimizer,lr=lr,tau=tau,b1=b1,b2=b2,momentum=momentum,meanshift=None)
        w = model.get_weights()
        self.loc_data = loc_data
        # Set initial distributions
        w_flat = torch.cat([w[k].flatten() for k in w])
        self.distribution = Normal(w_flat,torch.ones(len(w_flat)))

        # Set other parameters
        self.M=M
        self.swa_lr1=swa_lr1
        self.swa_lr2=swa_lr2
        self.swa_freq= swa_freq
        self.swa_epochs=swa_epochs
        self.verbose = verbose


    def combine(self, client_weights,device=None, client_scaling = None):
        # Dynamically calculate mean and variance of server weights.
        if self.verbose: print('FedBe: Reading in clients.')
        device = device if device!=None else 'cpu'
        mu_n = torch.zeros(self.model_size).to(device)
        s_n = torch.zeros(self.model_size).to(device)
        M_n = torch.zeros(self.model_size).to(device)
        S=[]
        for n,w_flat in enumerate(client_weights):
            mu_n2 = (n*mu_n+w_flat)/(n+1)
            M_n= M_n+(w_flat-mu_n)*(w_flat-mu_n2)
            if(n!=0):
                s_n= torch.sqrt(M_n/n)
            mu_n = mu_n2
            S.append(self._array_to_model_weight(w_flat))

        # Add small variance to avoid zero variance
        s_n = s_n+0.0000001
        self.distribution = Normal(mu_n,s_n)

        if self.verbose: print('FedBe: Creating ensamble of weights.')
        # Add mu
        S.append(self._array_to_model_weight(mu_n))
        # Sample new weights
        for i in range(self.M):
            w_s = self.distribution.sample()
            S.append(self._array_to_model_weight(w_s))

        ################ Evaluate on local data #######################
        if self.verbose: print('FedBe: Evaluating ensambles on local data.')
        loss = NLLLoss(reduction  ='sum')
        p=None
        X = []
        for w in S:
            self.model.set_weights(w)
            nll = self.model.evaluate(self.loc_data,loss_func=loss,take_mean=False,device=device)
            res = torch.exp(-torch.Tensor(nll))
            res = torch.nan_to_num(res, nan=0.0).to(device)
            if(p == None):
                p = torch.zeros(len(res)).to(device)
            p = p.add(res)
        p = p.div(len(S))

        ############## SWA ############################
        if self.verbose: print('FedBE: Running SWA')

        # initiate and run SWA
        self.model.set_weights(self._array_to_model_weight(mu_n))
        self.model.to(device)
        self.model.train()
        base_opt = torch.optim.SGD(self.model.parameters(), lr=self.swa_lr1)
        opt = SWA(base_opt, swa_start=5, swa_freq=self.swa_freq, swa_lr=self.swa_lr2)
        for i in range(self.swa_epochs):
             for j,(x,y) in enumerate(self.loc_data):
                 x = x.to(device)
                 y = y.to(device)
                 opt.zero_grad()
                 # To set average loss, we predict to set graph gradients
                 # Multiply the result by zero and add the average loss to get
                 # the propper backpropagation.
                 pred = self.model.predict(x)[0]
                 tmp_loss=(torch.mean(pred)*0-p[j])
                 tmp_loss.backward()
                 clip_grad_norm_(self.parameters(), 10)
                 opt.step()

        if self.verbose: print('FedBE: SWA Destilation done, updating model weights.')

        return self._model_weight_to_array(self.model.get_weights())
