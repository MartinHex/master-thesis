from Servers.ProbabilisticServer import ProbabilisticServer
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import random
import warnings
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from KDEpy.bw_selection import improved_sheather_jones

class FedKpServer(ProbabilisticServer):
    def __init__(self,model,shrinkage=1,store_distributions = False,cluster_mean=True,
                bandwidth = 'silverman',lr=1,tau=0.1,b1=.9,b2=0.99,momentum=1,meanshift=None,
                optimizer='none',max_iter=100,bandwidth_scaling=1, kernel_function = 'epanachnikov'):

        super().__init__(model,optimizer = optimizer,lr=lr,tau=tau,b1=b1,b2=b2,momentum=momentum)
        super().__init__(model)

        self.store_distributions = store_distributions
        self.max_iter=max_iter
        self.cluster_mean = cluster_mean
        self.bandwidth_scaling = bandwidth_scaling
        # Set bandwidth function
        self.bandwidths=None
        if(bandwidth =='plugin'):
            self.bandwidth_method = self._plugin
        elif(bandwidth=='crossval'):
            self.bandwidth_method = self._crossval
        elif(bandwidth=='local'):
            self.bandwidth_method = self._neirestneighbour
            self.nneigh = 5
        elif(bandwidth =='scott'):
            self.bandwidth_method = self._scott
        else:
            self.bandwidth_method = self._silverman

        # Set if adaptive bandwidth or not
        self.adaptive = (bandwidth in ['local'])

        # Set Kernel function
        assert kernel_function == 'gaussian' or kernel_function == 'epanachnikov', 'Specified kernel is not supported'
        if kernel_function=='epanachnikov':
            self.kernel=self._epanachnikov_kernel
        elif kernel_function=='gaussian':
            self.kernel=self._gaussian_kernel

        # Initiate weights and distribution
        if self.store_distributions:
            self.likelihood = [gaussian_kde([-1,0,1],1)for i in range(self.model_size)]
            self.stats = [[-1,1,0,1] for i in range(self.model_size)]
            self.prior = [gaussian_kde([-1,0,1],1) for i in range(self.model_size)]

        # Parameters for covariance adjustments
        self.beta = shrinkage
        self.shrinkage = shrinkage

    def combine(self, client_weights,device=None, client_scaling = None):

        # Kernel Esstimation
        if self.store_distributions:
            for i in range(self.model_size):
                x = client_weights[:,i]
                self.stats[i] = [torch.min(x),torch.max(x),torch.mean(x),torch.std(x)]
                self.prior[i] = self.likelihood[i]
                self.likelihood[i] = gaussian_kde(x,bw_method='silverman')

        # Calculate bandwidths
        if not self.adaptive:
            self.bandwidths = self.bandwidth_method(client_weights)*self.bandwidth_scaling

        # Mean shift algorithm:
        if self.cluster_mean:
            res_model_w = torch.zeros(self.model_size).to(device)
            for i,client_w in enumerate(client_weights):
                client_adj = self._mean_shift(client_weights,client_w,device=device)/len(client_weights)
                if(torch.any(torch.isnan(client_adj))): print('Nan found inclient adjusted weigth')
                res_model_w += client_adj
        else:
            mean_model = torch.mean(client_weights,0)
            res_model_w = self._mean_shift(client_weights,mean_model,device=device)

        return res_model_w

    def sample_model(self):
        if(self.store_distributions):
            w = [ker.resample(1)[0][0] for i,ker in enumerate(self.kernals)]
            return self._array_to_model_weight(w)
        else:
            print('Model does not have any distributions, change this by setting store_distributions=True')
            return self.get_weights()

    def _model_weight_to_array(self,w):
        return torch.cat([w[k].flatten() for k in w]).detach()

    def _array_to_model_weight(self,a):
        a_tens = torch.split(a,self.layer_size)
        model_w = {k:torch.reshape(a_tens[i],self.layer_shapes[i])
                    for i,k in enumerate(self.layers )}
        return model_w

    def plot_random_weights(self,shape,seed = 1234):
        random.seed(seed)
        if(self.store_distributions):
            nrow,ncol = shape
            size=nrow*ncol
            integers = [random.randint(0,self.model_size) for i in range(size)]
            fig, axs = plt.subplots(nrows=nrow,ncols=ncol,figsize=(ncol*3.2,nrow*3.2))
            for i,idx in enumerate(integers):
                self._plot_dist(idx,axs[i//ncol,i%ncol])
            plt.show()
        else:
            print('Model does not have any distributions, change this by setting store_distributions=True')

    def plot_many_layers(self,output_fldr,max=1000):
        if(self.store_distributions):
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
                    n_plots = min([max,max_idx-idx_sum])
                    for i in range(n_plots):
                        tmp_idx=i+idx_sum+1
                        self._plot_dist(tmp_idx,axs[i//ncol,i%ncol])
                    output_path = os.path.join(out_fldr,'%i-%i.jpg'%(idx_sum,idx_sum+i))
                    plt.savefig(output_path)
                    plt.clf()
                    plt.close()
                    idx_sum+=n_plots
                idx_sum = max_idx
        else:
            print('Model does not have any distributions, change this by setting store_distributions=True')

    def _plot_dist(self,idx,ax):
        ker = self.likelihood[idx]
        t_max = self.MLE_weight[idx]
        if(ker==None):
            ax.hist([t_max])
            return 0
        [min_x , max_x,mean_x,std_x] = self.stats[idx]
        t = torch.linspace(min_x-std_x,max_x+std_x,100)
        z = ker(t)
        ax.plot(t, z, '-')
        ax.fill(t, z, facecolor='blue', alpha=0.5)
        ax.set_xlim([min(t), max(t)])
        ax.plot(t_max,ker(t_max),'.',c='red',markersize=10)
        ax.plot([mean_x,mean_x],ax.get_ylim(),'--',c='black')

    def _mean_shift(self,client_weights,init,tol=1e-6,device=None):
        w_res = init.to(device)
        non_fixed_idx = torch.std(client_weights,0).nonzero().flatten()
        for i in range(self.max_iter):
            # Initiate parameters which are to be mean-shifted
            w = w_res[non_fixed_idx]
            n_nonzeros = len(w)
            if n_nonzeros==0: break
            if self.adaptive:
                client_w_tmp = client_weights[:,non_fixed_idx]
                H = self.bandwidth_method(client_w_tmp,x_0=w)*self.bandwidth_scaling
            else:
                H = self.bandwidths[non_fixed_idx]
            denominator= torch.zeros(n_nonzeros).to(device)
            numerator = torch.zeros(n_nonzeros).to(device)
            for _,client_w in enumerate(client_weights):
                w_i = client_w[non_fixed_idx].to(device)
                K = self.kernel((w-w_i)/H)
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
        if(i==self.max_iter-1):
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

    def _plugin(self,client_weights):
        bandwidths = []
        n = len(client_weights)
        for x in client_weights.transpose(0,1):
            try:
                h = improved_sheather_jones(x.reshape(len(x),1).numpy())
            except:
                std = torch.std(x)
                iqr = torch.quantile(x,0.75)-torch.quantile(x,0.25)
                h = n**(-0.2)*min(iqr,std)
            bandwidths.append(h)
        return torch.Tensor(bandwidths)

    def _crossval(self,client_weights):
        bandwidths = []
        n = len(client_weights)
        for x in client_weights.transpose(0,1):
            try:
                h_grid = (torch.std(x)*torch.logspace(0.01, 10, 4)).tolist()
                bandwidths = (torch.std(x)*torch.logspace(0.01, 10, 4)).tolist()
                grid = GridSearchCV(KernelDensity(),
                                    {'bandwidth': h_grid},
                                    cv=5)
                grid.fit(x.reshape(-1, 1))
                h =  h_grid[grid.best_index_]
            except:
                std = torch.std(x)
                iqr = torch.quantile(x,0.75)-torch.quantile(x,0.25)
                h = n**(-0.2)*min(iqr,std)
            bandwidths.append(h)
        return torch.Tensor(bandwidths)

    def _neirestneighbour(self,client_weights,x_0):
        se,_ = torch.sort((x_0-client_weights)**2,0)
        return self.nneigh**(-0.2)*torch.mean(se[:,:self.nneigh])
