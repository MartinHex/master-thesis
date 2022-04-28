from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Clients.SGDClient import SGDClient as Client
from Servers.FedKpServer import FedKpServer
from Servers.FedAvgServer import FedAvgServer
from Algorithms.FedAvg import FedAvg as Alg
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as grid_spec
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import os
from Models.Callbacks.Callbacks import Callbacks
import json

out_path  = os.path.join('data','Results','MNIST_exploration')
result_path = os.path.join(out_path,'results')
plot_path = os.path.join(out_path,'plots')
if not os.path.exists(out_path): os.mkdir(out_path)
if not os.path.exists(result_path): os.mkdir(result_path)
if not os.path.exists(plot_path): os.mkdir(plot_path)
# Parameters
alphas = [100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001]
n_clients = 10
seed = 0
batch_size = 16
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
dl = Dataloader(n_clients)
test_data = dl.get_test_dataloader(batch_size)

##################### Plot distributions based on alpha ######################
plt.style.use('seaborn-whitegrid')
torch.manual_seed(seed)
np.random.seed(seed)

fig,axs = plt.subplots(2,2,figsize=(6,6))
for i,alpha in enumerate([10,1,0.1,0.01]):
    ax = axs[i//2,i%2]
    dl = Dataloader(n_clients,alpha=alpha)
    for j,c_data in enumerate(dl.get_training_raw_data()):
        labels = [l for x,l in c_data]
        counts = np.zeros(10)
        for l in labels:
            counts[l]+=1
        counts = counts/5420
        ax.scatter(np.repeat(j,10)+1,np.arange(10)+1,s = 100*counts,color='blue')
    ax.set_title('alpha = %.2f'%alpha)
    ax.set_xticks(np.arange(10)+1)
    ax.set_yticks(np.arange(10)+1)
    ax.grid(color='grey',alpha=0.2)
    if(i%2==0):
        ax.set_ylabel('Labels')
    if(i//2==1):
        ax.set_xlabel('Client')

plt.savefig(os.path.join(plot_path,'sampledist.jpg'))
plt.show()


################### Study relation to Epochs ############################

# script
def test_script(rounds,n_clients,epochs):
    if not isinstance(rounds, list): rounds = np.repeat(rounds,4)
    if not isinstance(n_clients, list): n_clients = np.repeat(n_clients,4)
    if not isinstance(epochs, list): epochs = np.repeat(epochs,4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_model = Model()
    cbs = Callbacks(test_data,verbose=False,device=device)
    callbacks = [cbs.server_loss,cbs.server_accuracy,cbs.server_training_loss,cbs.server_training_accuracy,
                    cbs.skew,cbs.kurtosis,cbs.ks_test]
    res = pd.DataFrame(columns=['alpha','train_acc','train_loss','val_loss','val_acc',
                                'ks_test','kurtosis','skew','rounds','epochs','n_clients'])
    for alpha in alphas:
        idx = 0
        for round,n_client,epoch in zip(rounds,n_clients,epochs):
            dl = Dataloader(n_client,alpha=alpha)
            alg = Alg(dl,Model,batch_size=batch_size)
            alg.server.model.set_weights(init_model.get_weights())
            alg.run(round,epochs=epoch,callbacks=callbacks,device=device)
            r = alg.get_callback_data()
            train_acc = np.mean(r['server_training_accuracy'][-1])
            train_loss = np.mean(r['server_training_loss'][-1])
            val_loss = r['server_loss'][-1]
            val_acc = r['server_accuracy'][-1]
            for skew,kurtosis,ks in zip(r['skew'][-1],r['kurtosis'][-1],r['ks_test'][-1]):
                res = res.append({'alpha':alpha,'train_acc':train_acc,'train_loss':train_loss,
                                    'val_acc':val_acc,'val_loss':val_loss,
                                    'ks_test':ks,'kurtosis':kurtosis,
                                    'skew':skew,'rounds':round,'epochs':epoch,
                                    'n_clients':n_client},ignore_index = True)
            idx+=1
    return res



################# Res 1 ###############################################
rounds = 1
n_clients = 10
epochs = [1,2,5,10]
res1 = test_script(rounds=rounds,n_clients=n_clients,epochs=epochs)
res.to_csv(os.path.join(result_path,'res_epochs.csv'))

################# Res 2 ###############################################
rounds = [1,2,5,10]
n_clients = 10
epochs = 1
res = test_script(rounds=rounds,n_clients=n_clients,epochs=epochs)
res.to_csv(os.path.join(result_path,'res_rounds.csv'))

################# Res 3 ###############################################
rounds = 1
n_clients = [5,10,50,100]
epochs = 1
res3 = test_script(rounds=rounds,n_clients=n_clients,epochs=epochs)
res.to_csv(os.path.join(result_path,'res_n_clients.csv'))


################# Res 4 ###############################################

torch.manual_seed(seed)
np.random.seed(seed)
init_model = Model()
cbs = Callbacks(test_data,verbose=False,device=device)
callbacks = [cbs.server_loss,cbs.server_accuracy,cbs.server_training_loss,cbs.server_training_accuracy,
                cbs.skew,cbs.kurtosis,cbs.ks_test]
res = pd.DataFrame(columns=['alpha','train_acc','train_loss','val_loss','val_acc',
                            'ks_test','kurtosis','skew','rounds','epochs','n_clients'])

for alpha in alphas[1:]:
    dl = Dataloader(n_clients,alpha=alpha)
    alg = Alg(dl,Model,batch_size=batch_size)
    alg.server.model.set_weights(init_model.get_weights())
    alg.run(1,epochs=1,callbacks=callbacks,device=device)
    r = alg.get_callback_data()
    train_acc = np.mean(r['server_training_accuracy'][-1])
    train_loss = np.mean(r['server_training_loss'][-1])
    val_loss = r['server_loss'][-1]
    val_acc = r['server_accuracy'][-1]
    for skew,kurtosis,ks in zip(r['skew'][-1],r['kurtosis'][-1],r['ks_test'][-1]):
        res = res.append({'alpha':alpha,'train_acc':train_acc,'train_loss':train_loss,
                            'val_acc':val_acc,'val_loss':val_loss,
                            'ks_test':ks,'kurtosis':kurtosis,
                            'skew':skew,'rounds':1,'epochs':1,
                            'n_clients':n_clients},ignore_index = True)

import matplotlib.pyplot as plt
from matplotlib import gridspec  as grid_spec
from sklearn.neighbors import KernelDensity
import seaborn as sns
import os

res.to_csv(os.path.join(result_path,'res_alphas.csv'))

################# Res 5 ###############################################
seed=1
torch.manual_seed(seed)
np.random.seed(seed)
repeats = 5
init_model = Model()
cbs = Callbacks(test_data,verbose=False,device=device)
callbacks = [cbs.server_loss,cbs.server_accuracy,cbs.server_training_loss,cbs.server_training_accuracy]
res = pd.DataFrame(columns=['alpha','train_acc','train_loss','val_loss','val_acc'])
for _ in range(repeats):
    for alpha in alphas:
        dl = Dataloader(n_clients,alpha=alpha)
        alg = Alg(dl,Model,batch_size=batch_size)
        alg.server.model.set_weights(init_model.get_weights())
        alg.run(1,epochs=1,callbacks=callbacks,device=device)
        r = alg.get_callback_data()
        train_acc = np.mean(r['server_training_accuracy'][-1])
        train_loss = np.mean(r['server_training_loss'][-1])
        val_loss = r['server_loss'][-1]
        val_acc = r['server_accuracy'][-1]
        res = res.append({'alpha':alpha,'train_acc':train_acc,'train_loss':train_loss,
                            'val_acc':val_acc,'val_loss':val_loss},ignore_index = True)

res['val_acc'] = res['val_acc'].apply(lambda x: x.item())
res.to_csv(os.path.join(result_path,'res_lossacc.csv'))
res

########################################################################
############################ Plot Results ##############################
########################################################################


# Parameters
alphas = [10,1,0.1,0.01]
metrics =['skew','kurtosis','ks_test']
colors = sns.color_palette("ch:s=.25,rot=-.25", n_colors=4)

# Plot legend

fig, axs = plt.subplots(1,4, figsize=(4,10))

for i,(ax, alpha) in enumerate(zip(axs, alphas[::-1])):
    ax.imshow([[colors[i]]])
    name =  r'$\alpha='+'$%.2f'%alpha
    if i==3: name = r'$\alpha='+'$%.1f'%alpha
    ax.text(0.95, -0.1, name, va='center', ha='right', fontsize=10,
            transform=ax.transAxes)

# Turn off *all* ticks & spines, not just the ones with colormaps.
for ax in axs:
    ax.set_axis_off()

plt.savefig(os.path.join(plot_path,'legend.png'))



def plot_func(res,out_path,variable,metric):
    for j,var in enumerate(np.unique(res[variable])):
        fig = plt.figure(figsize=(4,4))
        gs = grid_spec.GridSpec(4,1,1)
        res_tmp = res[res[variable]==var]
        ax_objs=[]
        for i,alpha in enumerate(alphas[::-1]):
            ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
            ax = ax_objs[-1]
            res_tmp2 = res_tmp[res_tmp['alpha'] == alpha]
            x = np.array(res_tmp2[metric])
            acc = float(np.array(res_tmp2['val_acc'])[0].split('(')[-1][:-1])
            loss = np.array(res_tmp2['val_loss'])[0]
            x_d = np.linspace(np.min(x),np.max(x), 1000)

            kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
            kde.fit(x[:, None])

            logprob = kde.score_samples(x_d[:, None])

            # plotting the distribution
            ax.plot(x_d, np.exp(logprob),color="white",lw=1)
            ax.fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])


            # setting uniform x and y lims
            ax.set_xlim(np.min(x),np.max(x))

            # make background transparent
            rect = ax.patch
            rect.set_alpha(0)

            # remove borders, axis ticks, and labels
            ax.set_yticklabels([])
            ax.set_yticks([])

            if i == len(alphas)-1:
                ax.set_xlabel(metric, fontsize=8)
            else:
                ax.set_xticklabels([])
                ax.set_xticks([])

            spines = ["top","right","left","bottom"]
            for s in spines:
                ax.spines[s].set_visible(False)

        gs.update(hspace=-0.4)

        plt.tight_layout()
        name = '%s_%s_%i'%(metric,variable,var)
        plt.savefig(os.path.join(plot_path,name))

alphas = [10.  ,  1.  ,  0.1 ,  0.01]
######### Res 1 #############################
res = pd.read_csv(os.path.join(result_path,'res_epochs.csv'))
for metric in metrics:
    plot_func(res,out_path,'epochs',metric)

######### Res 2 #############################
res = pd.read_csv(os.path.join(result_path,'res_rounds.csv'))
for metric in metrics:
    plot_func(res,out_path,'rounds',metric)

######### Res 3 #############################
res = pd.read_csv(os.path.join(result_path,'res_n_clients.csv'))
for metric in metrics:
    plot_func(res,out_path,'n_clients',metric)

######## Res 4 ###############################
res = pd.read_csv(os.path.join(result_path,'res_alphas.csv'))
alphas = [100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001]
out_path = os.path.join('data','Results','Plots','Distributions')
metrics = ['ks_test','skew','kurtosis']
colors = sns.color_palette("ch:s=.25,rot=-.25", n_colors=len(alphas))
for metric in metrics:
    fig = plt.figure(figsize=(6,6))
    gs = grid_spec.GridSpec(len(alphas),1,1)
    ax_objs=[]
    for i,alpha in enumerate(alphas[::-1]):
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        ax = ax_objs[-1]
        res_tmp = res[res['alpha'] == alpha]
        x = np.array(res_tmp[metric])
        acc = np.array(res_tmp['val_acc'])[0]
        loss = np.array(res_tmp['val_loss'])[0]
        x_d = np.linspace(np.min(x),np.max(x), 1000)

        kde = KernelDensity(bandwidth=0.03, kernel='gaussian')
        kde.fit(x[:, None])

        logprob = kde.score_samples(x_d[:, None])

        # plotting the distribution
        ax.plot(x_d, np.exp(logprob),color="white",lw=1)
        ax.fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])

        # setting uniform x and y lims
        ax.set_xlim(np.min(x),np.max(x))

        # make background transparent
        rect = ax.patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax.set_yticklabels([])
        ax.set_yticks([])

        if i == len(alphas)-1:
            if metric == 'ks_test':
                ax.set_xlabel('p-value', fontsize=14)
            else:
                ax.set_xlabel(metric, fontsize=14)
        else:
            ax.set_xticklabels([])
            ax.set_xticks([])

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax.spines[s].set_visible(False)

        ax.text(np.min(x),0,r'$\alpha=$'+'%.3f'%(alpha),fontsize=10,ha="right")
        ax.grid(False)

    gs.update(hspace=-0.5)

    plt.tight_layout()

    name = 'alphas_%s'%(metric)
    plt.savefig(os.path.join(plot_path,name))

############## Res 5 #############################
res = pd.read_csv(os.path.join(result_path,'res_lossacc.csv')).drop('Unnamed: 0',axis=1)
stats = res.groupby('alpha').aggregate(['mean','std'])
cols = np.unique([c[0] for c in stats.columns])[[0,2,1,3]]
colors = sns.color_palette("Paired", n_colors=4)

fig,ax = plt.subplots(figsize=(6,6))
ax2 = ax.twinx()
x = np.array(stats.index)
for i,col in enumerate(cols):
    mu = np.array(stats[col]['mean'])
    std = np.array(stats[col]['std'])
    l = mu-std
    u = mu+std
    ax_tmp = ax2 if('loss' in col) else ax
    ax_tmp.plot(x,mu,label=col,color=colors[i])
    ax_tmp.fill_between(x,l,u,color=colors[i],alpha=0.3)
ax.legend(bbox_to_anchor=(1, -0.1),ncol=2,prop={'size': 10})
ax2.grid(False)
ax.grid(False)
ax2.legend(bbox_to_anchor=(0.5, -0.1),ncol=2,prop={'size': 10})
ax.set_xlabel(r'$\alpha$',fontsize=14)
ax.set_ylabel('Accuracy',fontsize=14)
ax2.set_ylabel('Loss',fontsize=14)
ax.set_xscale('log')

plt.savefig(os.path.join(plot_path,'lossandacc'))
plt.show()
