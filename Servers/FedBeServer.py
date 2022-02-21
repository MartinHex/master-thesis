from Servers.ABCServer import ABCServer
import torch
from torch.distributions import Normal
from torch.nn import NLLLoss
from torchcontrib.optim import SWA
from torch.optim import SGD
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class FedBeServer(ABCServer):

    def __init__(self,model,loc_data,M=10,swa_lr1=0.001,swa_lr2=0.0004,swa_batch_size=20,
                    swa_freq=25,seed=1,swa_epochs=1,verbose=True):
        super().__init__(model)
        w = model.get_weights()
        self.loc_data = loc_data
        # Set initial distributions
        w_flat = torch.cat([w[k].flatten() for k in w])
        self.distribution = Normal(w_flat,torch.ones(len(w_flat)))
        # Set tensorlengths for future reconstruction of flattening.
        self.tens_lengths = [len(w[k].flatten()) for k in w]
        self.model_shapes = [w[k].size() for k in w]

        # Set other parameters
        self.M=M
        self.swa_lr1=swa_lr1
        self.swa_lr2=swa_lr2
        self.swa_batch_size=swa_batch_size
        self.swa_freq= swa_freq
        self.swa_epochs=swa_epochs
        self.verbose = verbose
        self.seed = seed


    def aggregate(self, clients):
        # Dynamically calculate mean and variance of server weights.
        if self.verbose: print('FedBe: Reading in clients.')
        mu_n = 0
        s_n = 0
        M_n = 0
        S=[]
        for n,client in enumerate(clients):
            w = client.get_weights()
            w_flat = torch.cat([w[k].flatten() for k in w])
            mu_n2 = (n*mu_n+w_flat)/(n+1)
            M_n= M_n+(w_flat-mu_n)*(w_flat-mu_n2)
            if(n!=0):
                s_n= torch.sqrt(M_n/n)
            mu_n = mu_n2
            S.append(w)

        # Add small variance to avoid zero variance
        s_n = s_n+0.0000001
        self.distribution = Normal(mu_n,s_n)

        if self.verbose: print('FedBe: Creating ensamble of weights.')
        # Add mu
        mu_r = torch.split(mu_n,self.tens_lengths)
        mu_r = {k:torch.reshape(mu_r[i],self.model_shapes[i]) for i,k in enumerate(w)}
        S.append(mu_r)
        # Sample new weights
        for i in range(self.M):
            w_s = self.distribution.sample()
            w_r = torch.split(w_s,self.tens_lengths)
            w_r = {k:torch.reshape(w_r[i],self.model_shapes[i]) for i,k in enumerate(w)}
            S.append(w_r)

        ################ Evaluate on local data #######################
        if self.verbose: print('FedBe: Evaluating ensambles on local data.')
        loss = NLLLoss(reduction  ='sum')
        self.model.eval()
        p=np.array([])
        X = []
        for w in S:
            self.model.set_weights(w)
            res = []
            for x,y in self.loc_data:
                y_pred = self.model.forward(x)[0]
                # Calculate likelihood per sample
                for x,y_p,y_t in zip(x,y_pred,y):
                    nll =loss(y_p,y_t).detach()
                    l = torch.exp(-nll)
                    res.append(l)
                    # Add samples to map loss landscape only during first iteration
                    if(len(X)!=len(p)):
                        X.append(x)
            res = np.array(res)
            if(p.size == 0):
                p = np.zeros(len(res))
            p+=res

        p = p/len(S)
        T = [(x_j,p_j) for x_j,p_j in zip(X,p)]

        ############## SWA ############################
        if self.verbose: print('FedBE: Running SWA')
        # Batch data
        random.seed(self.seed)
        random.shuffle(T)
        T_batched = [T[i:((i+1)*self.swa_batch_size )] for i in range(len(T)//self.swa_batch_size)]

        # initiate and run SWA
        self.model.set_weights(mu_r)
        self.model.train()
        base_opt = torch.optim.SGD(self.model.parameters(), lr=self.swa_lr1)
        opt = SWA(base_opt,  swa_freq=self.swa_freq, swa_lr=self.swa_lr2)
        for i in range(self.swa_epochs):
             for batch in T_batched:
                 opt.zero_grad()
                 X_batched = [b[0] for b in batch]
                 p_batched = torch.tensor([b[1] for b in batch])
                 pred = self.model.forward(torch.stack(X_batched))[0]
                 loss=torch.mean(pred)*0-torch.mean(p_batched)
                 loss.backward()
                 opt.step()
             # if self.verbose: print('FedBE: Epoch %i: loss: %.4f'%(i,-loss.item()))

        if self.verbose: print('FedBE: SWA Destilation done, updating model weights.')
        opt.swap_swa_sgd()
