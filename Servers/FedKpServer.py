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
                bandwidth = 'silverman',lr=1,tau=0.1,b1=.9,b2=0.99,momentum=1,
                optimizer='none',max_iter=100,bandwidth_scaling=1, kernel_function = 'gaussian'):

        super().__init__(model,optimizer = optimizer,lr=lr,tau=tau,b1=b1,b2=b2,momentum=momentum)
        super().__init__(model)
        w = model.get_weights()
        self.layers = list(w)
        # Set tensorlengths for future reconstruction of flattening.
        self.layer_size = [len(w[k].flatten()) for k in w]
        self.model_size = sum(self.layer_size)
        self.layer_shapes = [w[k].size() for k in w]
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
        self.adaptive = (bandwidth=='local')

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

    def combine(self, clients,device=None, client_scaling = None):
        # List and translate data into numpy matrix
        client_weights = [self._model_weight_to_array(c.get_weights()) for c in clients]
        client_weights = torch.stack(client_weights).to(device)
        print('Aggregating Models.')

        # Kernel Esstimation
        if self.store_distributions:
            for i in range(self.model_size):
                x = client_weights[:,i]
                self.stats[i] = [torch.min(x),torch.max(x),torch.mean(x),torch.std(x)]
                self.prior[i] = self.likelihood[i]
                self.likelihood[i] = gaussian_kde(x,bw_method='silverman')

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

        res_model = self._array_to_model_weight(res_model_w.to('cpu'))
        return res_model

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
        if not self.adaptive: self._calc_bandwidth(client_weights)
        non_fixed_idx = torch.std(client_weights,0).nonzero().flatten()
        for i in range(self.max_iter):
            # Initiate parameters which are to be mean-shifted
            w = w_res[non_fixed_idx]
            n_nonzeros = len(w)
            if n_nonzeros==0: break
            if self.adaptive:
                client_w_tmp = client_weights[:,non_fixed_idx]
                H = self._local_bandwidth(client_w_tmp,x_0=w,device=device)*self.bandwidth_scaling
            else:
                H = self.bandwidths[non_fixed_idx]*self.bandwidth_scaling
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

        return w_res.to('cpu')

    def _gaussian_kernel(self,u):
        return torch.exp(-u**2/2)

    def _epanachnikov_kernel(self,u):
        u[torch.abs(u)>1] = 1
        return 3/4 * (1 - u**2)

    def _calc_bandwidth(self,client_weights,device=None):
        bandwidths = []
        for x in client_weights.transpose(0,1):
            h = self.bandwidth_method(x) if torch.std(x)!=0 else 0
            bandwidths.append(h)
        self.bandwidths = torch.Tensor(bandwidths).to(device)

    def _local_bandwidth(self,client_weights,x_0=None,device=None):
        bandwidths = []
        for i,x in enumerate(client_weights.transpose(0,1)):
            h = self.bandwidth_method(x,x_0[i]) if torch.std(x)!=0 else 0
            bandwidths.append(h)
        return torch.Tensor(bandwidths).to(device)

    def _scott(self,x):
        n = len(x)
        sig = torch.std(x)
        iqr = torch.quantile(x,0.75)-torch.quantile(x,0.5)

        return n**(-0.2)*torch.min(sig,iqr)

    def _silverman(self,x):
        n = len(x)
        sig = torch.std(x)
        iqr = torch.quantile(x,0.75)-torch.quantile(x,0.5)

        return (n*3/4)**(-1/5)*torch.min(sig,iqr)

    def _plugin(self,x):
        try:
            h = improved_sheather_jones(x.reshape(len(x),1).numpy())
        except:
            h = self._scott(x)
        return h

    def _crossval(self,x):
        bandwidths = (torch.std(x)*torch.logspace(0.01, 10, 4)).tolist()
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': bandwidths},
                            cv=5)
        grid.fit(x.reshape(-1, 1))
        return bandwidths[grid.best_index_]

    def _neirestneighbour(self,x,x_0):
        se,_ = torch.sort((x_0-x)**2)
        return self.nneigh**(-0.2)*torch.mean(se[:self.nneigh])
