from Servers.ProbabilisticServer import ProbabilisticServer
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import random
import warnings

class FedKPServer(ProbabilisticServer):
    def __init__(self,model,shrinkage=1,store_distributions = False):
        super().__init__(model)
        w = model.get_weights()
        self.layers = list(w)
        # Set tensorlengths for future reconstruction of flattening.
        self.layer_size = [len(w[k].flatten()) for k in w]
        self.model_size = sum(self.layer_size)
        self.layer_shapes = [w[k].size() for k in w]
        self.store_distributions = store_distributions

        # Initiate weights and distribution
        self.MLE_weight = torch.zeros(self.model_size)
        if self.store_distributions:
            self.likelihood = [gaussian_kde([-1,0,1],1)for i in range(self.model_size)]
            self.stats = [[-1,1,0,1] for i in range(self.model_size)]
            self.prior = [gaussian_kde([-1,0,1],1) for i in range(self.model_size)]

        # Parameters for covariance adjustments
        self.beta = shrinkage
        self.shrinkage = shrinkage

    def aggregate(self, clients,cov_adj = True):
        # List and translate data into numpy matrix
        client_weights = [c.get_weights() for c in clients]
        client_weights = torch.stack([self._model_weight_to_array(c.get_weights())for c in clients])

        if(cov_adj):
            client_weights = self._cov_adj_client_weights(client_weights)
            client_weights = torch.stack(client_weights)
        # Kernel Esstimation
        if self.store_distributions:
            for i in range(self.model_size):
                x = client_weights[:,i]
                self.stats[i] = [torch.min(x),torch.max(x),torch.mean(x),torch.std(x)]
                self.likelihood[i] = gaussian_kde(x,bw_method='silverman')

        # Mean shift algorithm:
        self.MLE_weight = self._mean_shift(client_weights)

        res_model = self._array_to_model_weight(self.MLE_weight)
        self.model.set_weights(res_model)

    def sample_model(self):
        if(self.store_distributions):
            w = [ker.resample(1)[0][0] if ker!= None else self.MLE_weight[i] for i,ker in enumerate(self.kernals)]
            return self._array_to_model_weight(w)
        else:
            print('Model does not have any distributions, change this by setting store_distributions=True')
            return self.MLE_weight()

    def _model_weight_to_array(self,w):
        return torch.cat([w[k].flatten() for k in w]).detach()

    def _array_to_model_weight(self,a):
        a_tens = torch.split(a,self.layer_size)
        model_w = {k:torch.reshape(a_tens[i],self.layer_shapes[i])
                    for i,k in enumerate(self.layers )}
        return model_w


    def plot_random_weights(self,shape):
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

    def _cov_adj_client_weights(self, client_weights):
        mean_w = torch.mean(client_weights,0)
        res_w = [mean_w.detach() for i in range(len(client_weights))]
        for i,w_0 in enumerate(client_weights):
            w = [cw for j,cw in enumerate(client_weights) if j!=i]
            cov_adj_w = self._cov_adj_weight(w,w_0)
            res_w[i] = res_w[i].add(cov_adj_w)
        return res_w

    def _cov_adj_weight(self, w,w_0):
        current_delta = w_0.sub(w[0])
        current_average = w_0.add(w[0]).div(2)
        device = w[0].get_device()
        u = [torch.zeros(self.model_size).to(device)]
        v = [torch.zeros(self.model_size).to(device)]
        sum = torch.zeros(self.model_size).to(device)
        # Produce the desired gradient online and at any-time as described in appendix C by Al-Shedivat et al.
        # (https://arxiv.org/pdf/2010.05273.pdf)
        for i in range(1,len(w)):
            t=i+1
            u.append(w[i].sub(current_average))
            for k in range(i - 1):
                gamma_k = (self.beta * k) / (k + 1)
                nominator = gamma_k * u[-1].matmul(v[k]).item()
                denominator = 1 + gamma_k * u[k].matmul(v[k]).item()
                sum = sum.add(v[k].mul(nominator / denominator))
            v.append(u[i].sub(sum))
            current_average = w[i].add(current_average,alpha=i).div(i + 1)
            gamma_t = (self.beta * (t - 1)) / t
            nominator = gamma_t * (t * u[i].matmul(current_delta) -u[i].matmul(v[i]))
            denominator = 1 + gamma_t * v[i].matmul(u[i])
            current_delta = current_delta.sub((1 + nominator / denominator).mul(v[i]).div(i + 1))
            current_average = w[i]+(i*current_average)/(i + 1)

        new_weights = w_0.sub(current_delta.div(self.shrinkage))
        return new_weights

    def _mean_shift(self,client_weights,tol=0.00001,max_iter = 100):
        w = torch.mean(client_weights,0).reshape(1,self.model_size)
        dif = tol+ 1
        i = 0
        while dif>tol and i<max_iter:
            dist = torch.cdist(w,client_weights)
            exp_dist = torch.exp(-dist.div(self.model_size))
            denominator = torch.sum(exp_dist)
            m_x = exp_dist.matmul(client_weights).div(denominator)
            dif =torch.cdist(w,m_x).item()
            w = m_x
            i+=1
        if(i>=max_iter):
            warnings.warn("Maximal iteration reacher. You may want to look into increasing the amount of iterations.")
        return w[0]
