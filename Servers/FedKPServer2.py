from Servers.ABCServer import ABCServer
import torch
from torch.distributions import Normal
from scipy.stats import gaussian_kde
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt

class FedKPServer(ABCServer):
    def __init__(self,model,shrinkage=1):
        super().__init__(model)
        w = model.get_weights()
        self.layers = list(w)
        # Set tensorlengths for future reconstruction of flattening.
        self.layer_size = [len(w[k].flatten()) for k in w]
        self.model_size = sum(self.layer_size)
        self.layer_shapes = [w[k].size() for k in w]

        # Initiate weights and distribution
        self.MLE_weight = np.zeros(self.model_size)
        self.likelihood = [gaussian_kde([-1,0,1],1)for i in range(self.model_size)]
        self.stats = [[-1,1,0,1] for i in range(self.model_size)]
        self.prior = [gaussian_kde([-1,0,1],1) for i in range(self.model_size)]

        # Parameters for covariance adjustments
        self.beta = shrinkage
        self.shrinkage = shrinkage

    def aggregate(self, clients,cov_adj = True):
        # List and translate data into numpy matrix
        client_weights = [c.get_weights() for c in clients]
        client_weights = np.array([self._model_weight_to_array(c.get_weights())
                                    for c in clients])

        if(cov_adj):
            client_weights = self._cov_adj_client_weights(client_weights)
        # Kernel Esstimation
        for i in range(self.model_size):
            x = client_weights[:,i]
            self.stats[i] = [np.min(x),np.max(x),np.mean(x),np.std(x)]
            self.likelihood[i],self.MLE_weight[i] = self._kernelEsstimator(x)

        res_model = self._array_to_model_weight(self.MLE_weight)
        self.model.set_weights(res_model)

    def _kernelEsstimator(self,x):
        try:
            ker = gaussian_kde(x,bw_method='silverman')
        except:
            return None,x[0]
        z = ker(x)
        t_0= x[np.argmax(z)]
        try:
            t_max = minimize(lambda t:-ker(t),t_0,method='BFGS',gtol=0.000001).x[0]
        except:
            t_max = t_0
        return ker,t_max

    def sample_model(self):
        w = [ker.resample(1)[0][0] for ker in self.likelihood]
        return self._array_to_model_weight(w)

    def _model_weight_to_array(self,w):
        return torch.cat([w[k].flatten() for k in w]).detach().numpy()

    def _array_to_model_weight(self,a):
        a_tens = torch.Tensor(a)
        a_tens = torch.split(a_tens,self.layer_size)
        model_w = {k:torch.reshape(a_tens[i],self.layer_shapes[i])
                    for i,k in enumerate(self.layers )}
        return model_w


    def plot_random_weights(self,shape):
        nrow,ncol = shape
        integers = np.random.randint(self.model_size,size=nrow*ncol)
        fig, axs = plt.subplots(nrows=nrow,ncols=ncol,figsize=(ncol*3.2,nrow*3.2))
        for i,idx in enumerate(integers):
            self.plot_dist(idx,axs[i//ncol,i%ncol])
        plt.show()

    def plot_many_layers(self,output_fldr,max=1000):
        nrow = 10
        ncol = 10
        size = nrow*ncol
        idx_sum = 0
        if not os.path.exists(output_fldr):os.mkdir(output_fldr)
        for length,k in zip(self.layer_size,self.layers):
            out_fldr = os.path.join(output_fldr,k)
            if not os.path.exists(out_fldr):os.mkdir(out_fldr)
            max_idx = idx_sum + min(length,max)
            while idx_sum<max_idx:
                fig, axs = plt.subplots(nrows=nrow,ncols=ncol)
                n_plots = np.min([max,max_idx-idx_sum])
                for i in range(n_plots):
                    tmp_idx=i+idx_sum+1
                    self.plot_dist(tmp_idx,axs[i//ncol,i%ncol])
                output_path = os.path.join(out_fldr,'%i-%i.jpg'%(idx_sum,idx_sum+i))
                plt.savefig(output_path)
                plt.clf()
                plt.close()
                idx_sum+=n_plots
            idx_sum = max_idx

    def plot_dist(self,idx,ax):
        ker = self.likelihood[idx]
        t_max = self.MLE_weight[idx]
        if(ker==None):
            ax.hist([t_max])
            return 0
        [min_x , max_x,mean_x,std_x] = self.stats[idx]
        t = np.linspace(min_x-std_x,max_x+std_x,100)
        z = ker(t)
        ax.plot(t, z, '-')
        ax.fill(t, z, facecolor='blue', alpha=0.5)
        ax.set_xlim([np.min(t), np.max(t)])
        ax.plot(t_max,ker(t_max),'.',c='red',markersize=10)
        ax.plot([mean_x,mean_x],ax.get_ylim(),'--',c='black')

    def _cov_adj_client_weights(self, client_weights):
        mean_w = np.mean(client_weights,axis=0)
        res_w = np.array([mean_w for i in range(len(client_weights))])
        for i,w_0 in enumerate(client_weights):
            w = [cw for j,cw in enumerate(client_weights) if j!=i]
            cov_adj_w = self._cov_adj_weight(w,w_0)
            res_w[i] = res_w[i]+cov_adj_w
        return res_w

    def _cov_adj_weight(self, w,w_0):
        current_delta = w_0-(w[0])
        current_average = w_0+(w[0])/2
        u = [np.zeros(self.model_size)]
        v = [np.zeros(self.model_size)]
        sum = 0
        # Produce the desired gradient online and at any-time as described in appendix C by Al-Shedivat et al.
        # (https://arxiv.org/pdf/2010.05273.pdf)
        for i in range(1,len(w)):
            t=i+1
            u.append(w[i]-(current_average))
            for k in range(i - 1):
                gamma_k = (self.beta * k) / (k + 1)
                nominator = gamma_k * np.matmul(u[-1],v[k])
                denominator = 1 + gamma_k * np.matmul(u[k],v[k])
                sum+=(nominator / denominator)* v[k]
            v.append(u[i]-sum)
            current_average = w[i]+(i*current_average)/(i + 1)
            gamma_t = (self.beta * (t - 1)) / t
            nominator = gamma_t * (t * np.matmul(u[i],current_delta) -np.matmul( u[i],v[i]))
            denominator = 1 + gamma_t * np.matmul(v[i],u[i])
            current_delta = current_delta-((1 + nominator / denominator)*(v[i])/(i + 1))
            current_average = w[i]+(i*current_average)/(i + 1)

        new_weights = w_0-(current_delta / self.shrinkage)
        return new_weights

np.mean(np.zeros((5,4)),axis=0)
x = np.ones(5)
x.T
from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Clients.SGDClient import SGDClient
import torch
np.matmul(x,x.T)
n_clients = 100
dl = Dataloader(100)
clients = [SGDClient(Model(),d) for d in dl.get_training_dataloaders(16)]
server = FedKPServer(Model())

server.push_weights(clients)
for client in clients:
    client.train()

server.aggregate(clients)
