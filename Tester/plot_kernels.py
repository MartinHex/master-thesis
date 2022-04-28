from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Clients.SGDClient import SGDClient as Client
from Servers.FedKpServer import FedKpServer
from Servers.FedAvgServer import FedAvgServer
from Algorithms.FedKp import FedKp as Alg
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import os
from Models.Callbacks.Callbacks import Callbacks
import json
from sklearn.neighbors import KernelDensity
import scipy.stats as stats

# Parameters
out_path = os.path.join('data','Results','kernel_plots')
n_clients = 25
seed = 0
batch_size = 16

# Variables
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
dl = Dataloader(n_clients,alpha=alpha)
test_data = dl.get_test_dataloader(batch_size)
clients = [Client(Model(),l) for l in dl.get_training_dataloaders(batch_size)]

# Setup
if not os.path.exists(out_path): os.mkdir(out_path)

def model_weight_to_array(w):
    return torch.cat([w[k].flatten() for k in w]).detach()

def array_to_model_weight(a):
    a_tens = torch.split(a,self.layer_size)
    model_w = {k:torch.reshape(a_tens[i],self.layer_shapes[i])
                for i,k in enumerate(self.layers )}
    return model_w

def kernel_denisty(samples,x):
    std = np.std(samples)
    iqr = np.quantile(samples,0.75)-np.quantile(samples,0.25)
    h = np.min([std,iqr])*len(samples)**(-0.2)*1.33
    kde = KernelDensity(bandwidth=h, kernel='epanechnikov')
    kde.fit(samples[:, None])
    logprob = kde.score_samples(x[:, None])
    return np.exp(logprob)

################ Plot kernel density example ########################
n_samples = 100
x = np.linspace(-3,3,1000)

fig,axs = plt.subplots(1,3,figsize=(12,3.5))

# Normal example
ax = axs[0]
y_dist = stats.norm.pdf(x, 0, 1)
samples = stats.norm.rvs(size=n_samples)
y_ker = kernel_denisty(samples,x)
ax.plot(x,y_dist,color='black')
ax.fill_between(x, y_ker, alpha=1,color=colors[10])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Normal Distribution')

# Bimodal example
ax = axs[1]
y_dist = (stats.norm.pdf(x, 1.5, 0.5)+stats.norm.pdf(x, -1.5, 0.5))/2
samples = stats.norm.rvs(-1.5,0.5,size=n_samples//2)
samples = np.append(samples,stats.norm.rvs(1.5,0.5,size=n_samples//2))
y_ker = kernel_denisty(samples,x)
ax.plot(x,y_dist,color='black',label='True Distribution')
ax.fill_between(x, y_ker, alpha=1,color=colors[10],label='Kernel Esstimate')
ax.set_xlabel('x')
ax.set_title('Bimodal Distribution')
ax.legend(bbox_to_anchor=(1.15, -0.2),ncol=2)

# Skewed distribution
ax = axs[2]
y_dist = stats.skewnorm.pdf(x,5, 0, 1)
samples = stats.skewnorm.rvs(5,size=n_samples)
y_ker = kernel_denisty(samples,x)
ax.plot(x,y_dist,color='black',label='True Distribution')
ax.fill_between(x, y_ker, alpha=1,color=colors[10],label='Kernel Esstimate')
ax.set_xlabel('x')
ax.set_title('Skewed Distribution')

plt.savefig(os.path.join(out_path,'KernelExample'))
plt.show()


################# Plot random example ################################
alphas = [10,1,0.1,0.01]
torch.manual_seed(seed)
np.random.seed(seed)
init_model = Model()
randindx =  np.random.randint(0,10)
randindx

fig,axs = plt.subplots(1,4,figsize=(16,3.5))
for alpha,ax in zip(alphas,axs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dl = Dataloader(n_clients,alpha=alpha)
    clients = [Client(Model(),l) for l in dl.get_training_dataloaders(batch_size)]

    # Setup
    for client in clients:
        client.set_weights(init_model.get_weights())
        client.train()
    client_weights = [c.get_weights()['out.bias'].flatten() for c in clients]
    client_weights = torch.stack(client_weights).to(device)
    x = client_weights[:,randindx].numpy()
    h = np.std(x)*len(x)**(-0.2)
    x_d = np.linspace(np.min(x)-1.5*h,np.max(x)+1.5*h, 1000)
    mean_x = np.mean(x)

    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(x[:, None])

    logprob = kde.score_samples(x_d[:, None])

    # plotting the distribution
    ax.plot(x_d, np.exp(logprob),color="white",lw=1.5)
    ax.fill_between(x_d, np.exp(logprob), alpha=1,color=colors[10])
    ax.set_xlabel('x')
    ax.plot([mean_x,mean_x],ax.get_ylim(),'--',c='black')
    if alpha==10:
        ax.set_ylabel('w')
    ax.set_title('alpha = %s'%str(alpha))

plt.savefig(os.path.join(out_path,'MNIST_alphas.png'))
plt.show()


########################### Plot title image ############################
alpha=0.1
dl = Dataloader(n_clients,alpha=alpha)
clients = [Client(Model(),l) for l in dl.get_training_dataloaders(batch_size)]
torch.manual_seed(seed)
np.random.seed(seed)
init_model = Model()

for client in clients:
    client.set_weights(init_model.get_weights())
    client.train()

client_weights = [model_weight_to_array(c.get_weights()) for c in clients]
client_weights = torch.stack(client_weights).to(device)

n_weights = 25
integers = [np.random.randint(0,len(client_weights[0])) for i in range(n_weights)]

fig = plt.figure(figsize=(5,5))
colors = sns.color_palette("ch:s=.25,rot=-.25", n_colors=n_weights)
gs = grid_spec.GridSpec(n_weights,1,1)
ax_objs=[]
for i,randindx in enumerate(integers):
    ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
    ax = ax_objs[-1]
    x = client_weights[:,randindx].numpy()
    h = np.std(x)*len(x)**(-0.2)*2
    x_d = np.linspace(np.min(x)-h,np.max(x)+h, 1000)

    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(x[:, None])

    logprob = kde.score_samples(x_d[:, None])

    # plotting the distribution
    ax.plot(x_d, np.exp(logprob),color="white",lw=1.5)
    ax.fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])

    # setting uniform x and y lims
    ax.set_xlim(np.min(x),np.max(x))

    # make background transparent
    rect = ax.patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])

    spines = ["top","right","left","bottom"]
    for s in spines:
        ax.spines[s].set_visible(False)

gs.update(hspace=-0.8)

plt.tight_layout()
plt.savefig(os.path.join(out_path,'title'))
plt.show()
