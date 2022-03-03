from Servers.ProbabilisticServer import ProbabilisticServer
import torch
from torch.distributions import Normal
from scipy.stats import gaussian_kde
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt


class FedKPServer(ProbabilisticServer):
    def __init__(self,model):
        super().__init__(model)
        w = model.get_weights()
        self.layers = list(w)
        # Set tensorlengths for future reconstruction of flattening.
        self.layer_size = [len(w[k].flatten()) for k in w]
        self.model_size = sum(self.layer_size)
        self.layer_shapes = [w[k].size() for k in w]

        # Initiate weights and distribution
        self.MLE_weight = np.zeros(self.model_size)
        self.kernals = [gaussian_kde([0,1],bw_method='silverman') for i in range(self.model_size)]
        self.stats = [[-1,1,0,1] for i in range(self.model_size)]

    def aggregate(self, clients,cov_adj=False,select_MLE=False):
        # List and translate data into numpy matrix
        client_weights = np.array([self._model_weight_to_array(c.get_weights())
                                    for c in clients],dtype='float64')

        if(cov_adj):
            sig = np.cov(client_weights.T)
            # Stabalize esstimate
            sig = sig+0.001*np.eye(sig.shape[0])
            sig_inv = np.linalg.inv(sig)
            client_weights = np.matmul(sig_inv,client_weights.T).T
        # Kernel Esstimation
        for i in range(self.model_size):
            x = client_weights[:,i]
            self.stats[i] = [np.min(x),np.max(x),np.mean(x),np.std(x)]
            self.kernals[i],self.MLE_weight[i] = self._kernelEsstimator(x)

        if(cov_adj):
            self.MLE_weight = np.matmul(sig,self.MLE_weight.T)

        if(select_MLE):
            res_model = self._array_to_model_weight(self.MLE_weight)
        else:
            res_model = self.sample_model()
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
        w = [ker.resample(1)[0][0] for ker in self.kernals]
        return self._array_to_model_weight(w)

    def _model_weight_to_array(self,w):
        return torch.cat([w[k].flatten() for k in w]).cpu().detach().numpy()

    def _array_to_model_weight(self,a):
        a_tens = torch.Tensor(a)
        a_tens = torch.split(a_tens,self.layer_size)
        model_w = {k:torch.reshape(a_tens[i],self.layer_shapes[i])
                    for i,k in enumerate(self.layers )}
        return model_w


    def plot_random_weights(self,shape,seed=None):
        nrow,ncol = shape
        if(seed!=None):
            np.random.seed(seed)
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
        ker = self.kernals[idx]
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
