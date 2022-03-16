from Servers.ProbabilisticServer import ProbabilisticServer
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import warnings

class FedKpServer(ProbabilisticServer):
    def __init__(self,model,shrinkage=1,store_distributions = False,cov_adj = False,
                bandwidth = 'silverman',lr=1,tau=0.1,b1=.9,b2=0.99,momentum=1,
                optimizer='none'):

        super().__init__(model,optimizer = optimizer,lr=lr,tau=tau,b1=b1,b2=b2,momentum=momentum)
        super().__init__(model)
        w = model.get_weights()
        self.layers = list(w)
        # Set tensorlengths for future reconstruction of flattening.
        self.layer_size = [len(w[k].flatten()) for k in w]
        self.model_size = sum(self.layer_size)
        self.layer_shapes = [w[k].size() for k in w]
        self.cov_adj = cov_adj
        self.store_distributions = store_distributions
        # Set bandwidth function
        if(bandwidth =='silverman'):
            self.bandwidth_method = self._silverman
        if(bandwidth =='scott'):
            self.bandwidth_method = self._scott

        # Initiate weights and distribution
        self.MLE_weight = torch.zeros(self.model_size)
        if self.store_distributions:
            self.likelihood = [gaussian_kde([-1,0,1],1)for i in range(self.model_size)]
            self.stats = [[-1,1,0,1] for i in range(self.model_size)]
            self.prior = [gaussian_kde([-1,0,1],1) for i in range(self.model_size)]

        # Parameters for covariance adjustments
        self.beta = shrinkage
        self.shrinkage = shrinkage

    def combine(self, clients,device=None):
        # List and translate data into numpy matrix
        client_weights = [c.get_weights() for c in clients]
        client_weights = torch.stack([self._model_weight_to_array(c.get_weights())for c in clients])
        print('Aggregating Models.')
        if(self.cov_adj):
            client_weights = self._cov_adj_client_weights(client_weights,device=device)
            client_weights = torch.stack(client_weights)
        # Kernel Esstimation
        if self.store_distributions:
            for i in range(self.model_size):
                x = client_weights[:,i]
                self.stats[i] = [torch.min(x),torch.max(x),torch.mean(x),torch.std(x)]
                self.likelihood[i] = gaussian_kde(x,bw_method='silverman')

        # Mean shift algorithm:
        self.MLE_weight = self._mean_shift(client_weights,device=device)

        res_model = self._array_to_model_weight(self.MLE_weight)
        return res_model

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

    def _cov_adj_client_weights(self, client_weights,device=None):
        mean_w = torch.mean(client_weights,0)
        res_w = [mean_w.detach() for i in range(len(client_weights))]
        for i,w_0 in enumerate(tqdm(client_weights)):
            w = [cw for j,cw in enumerate(client_weights) if j!=i]
            cov_adj_w = self._cov_adj_weight(w,w_0,device=device)
            res_w[i] = res_w[i].add(cov_adj_w)
        return res_w

    def _cov_adj_weight(self, w,w_0,device=None):
        device = device if device !=None else 'cpu'
        w_0 = w_0.to(device)
        current_delta = w_0.sub(w_0).to(device)
        current_average = w[1].sub(w[0]).div(2).to(device)
        u_t = w[1].sub(w[0]).to(device)
        v = [u_t]
        gamma_factors = []
        # Produce the desired gradient online and at any-time as described in appendix C by Al-Shedivat et al.
        # (https://arxiv.org/pdf/2010.05273.pdf)
        for i in range(2,len(w)):
            w_i = w[i].to(device)
            u_t = w_i.sub(current_average)
            t=i
            # Add to sum
            gamma_k = (self.beta * i) / (i + 1)
            nominator = gamma_k
            denominator = 1 + gamma_k * u_t.matmul(v[-1]).item()
            gamma_factors.append(gamma_k/denominator)
            sum = torch.zeros(self.model_size).to(device)
            for i,v_k in enumerate(v):
                alpha = gamma_factors[i]+torch.matmul(v_k,u_t)
                sum = sum.add(v_k,alpha=alpha)

            # Update current weights
            v_t = u_t.sub(sum)
            v.append(v_t)
            # Calculate adjusted delta
            current_average = w_i.add(current_average,alpha=i).div(i + 1)
            gamma_t = (self.beta * (t - 1)) / t
            nominator = gamma_t * (t * u_t.matmul(current_delta) -u_t.matmul(v_t))
            denominator = 1 + gamma_t * v_t.matmul(u_t)
            current_delta = current_delta.sub((1 + nominator / denominator).mul(v_t).div(i + 1))

        new_weights = w_0.sub(current_delta.div(self.shrinkage)).to('cpu')
        return new_weights

    def _mean_shift(self,client_weights,tol=0.000001,max_iter = 10000,device=None):
        H = self.bandwidth_method(client_weights,device=device)
        w = torch.mean(client_weights,0).to(device)
        #print(w)
        dif = tol+ 1
        i = 0
        while i<10:
            denominator= torch.zeros(self.model_size).to(device)
            numerator = torch.zeros(self.model_size).to(device)
            for _,client_w in enumerate(client_weights):
                w_i = client_w.to(device)
                dif_tmp = (w-w_i)/H
                dist = dif_tmp**2
                exp_dist = torch.exp(-dist)
                denominator += exp_dist
                numerator += exp_dist*w_i
            m_x = numerator/(denominator)
            dif =torch.mean(torch.abs(w-m_x))
            w = torch.clone(m_x)
            i+=1
        if(i>=max_iter):
            warnings.warn("Maximal iteration reacher. You may want to look into increasing the amount of iterations.")
        #print(w)
        return w.to('cpu')

    def _scott(self,client_weights,device=None):
        n = len(client_weights)
        sig = torch.std(client_weights,0).to(device)
        d = len(client_weights[0])

        return n**(-1/(d+4))*sig+0.000001


    def _silverman(self,client_weights,device=None):
        n = len(client_weights)
        sig = torch.std(client_weights,0).to(device)
        d = len(client_weights[0])

        return (4/(d+2))**(1/(d+4))*n**(-1/(d+4))*sig+0.000001
