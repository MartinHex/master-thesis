import sys
sys.path.append('.')
from Models.MNIST_Model import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
from Algorithms.FedAvg import FedAvg
from Algorithms.FedKp import FedKp
from Models.Callbacks.Callbacks import Callbacks
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as grid_spec
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm

# Specify out_path
out_path  = os.path.join('data','Results','MNIST','Hyperparameter_evaluation')
log_path = os.path.join(out_path,'results')
plot_path = os.path.join(out_path,'plots')
if not os.path.exists(out_path): os.mkdir(out_path)
if not os.path.exists(log_path): os.mkdir(log_path)
if not os.path.exists(plot_path): os.mkdir(plot_path)

# Parameters
alphas = [10,1,0.1,0.01]
n_clients = 100
seed = 0
batch_size = 16
n_runs = 5

# Variables
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
# Initiate test data before running everything
dl = Dataloader(n_clients)
test_data = dl.get_test_dataloader(batch_size)

# Initiate Algorithms
print('Creating FedAvg')
torch.manual_seed(seed)
np.random.seed(seed)
fedAvg = FedAvg(dl,Model,batch_size=batch_size,clients_per_round=10)
print('Creating FedKp')
torch.manual_seed(seed)
np.random.seed(seed)
fedKp = FedKp(dl,Model,batch_size=batch_size,cluster_mean=False,clients_per_round=10)
torch.manual_seed(seed)
np.random.seed(seed)
fedKp_clstr = FedKp(dl,Model,batch_size=batch_size,cluster_mean=True,clients_per_round=10)

alghs = {
    'FedAvg':fedAvg,
    'FedKp':fedKp,
    'FedKP_cluter_mean':fedKp_clstr
}

# script
def test_script(rounds,clients_per_rounds,epochs):
    # Initiate all hyperparameters to lists
    if not isinstance(rounds, list): rounds = np.repeat(rounds,4)
    if not isinstance(clients_per_rounds, list): clients_per_rounds = np.repeat(clients_per_rounds,4)
    if not isinstance(epochs, list): epochs = np.repeat(epochs,4)
    # Set seed to being the same
    torch.manual_seed(seed)
    np.random.seed(seed)
    init_model = Model()
    cbs = Callbacks(test_data,verbose=False,device=device)
    callbacks = [   cbs.server_thesis_results,
                    cbs.server_training_thesis_results,
                    cbs.skew,cbs.kurtosis,cbs.ks_test]
    alg_res = pd.DataFrame(columns=['algorithm','alpha','run',
                                'train_loss','train_acc','train_rec','train_prec'
                                'val_loss','val_acc','val_rec','val_prec'
                                'ks_test','kurtosis','skew','rounds','epochs','clients_per_round'])

    dist_res = pd.DataFrame(columns=['algorithm','alpha','run',
                                'ks_test','kurtosis','skew','rounds','epochs','clients_per_round'])

    for alpha in alphas:
        print('alpha=%s'%str(alpha))
        for run in range(n_runs):
            dl = Dataloader(n_clients,alpha=alpha)
            for alg in alghs:
                alghs[alg].dataloaders = dl.get_training_dataloaders(batch_size)
            for round,clients_per_round,epoch in zip(rounds,clients_per_rounds,epochs):
                for alg in alghs:
                    # Train algorithm
                    alghs[alg].clients_per_round = clients_per_round
                    alghs[alg].server.model.set_weights(init_model.get_weights())
                    alghs[alg].run(round,epochs=epoch,callbacks=callbacks,device=device,log_callbacks=False)

                    # Evaluate algorithm
                    r = alghs[alg].get_callback_data()
                    train_loss = np.mean(r['server_training_loss'][-1])
                    train_acc = np.mean(r['server_training_accuracy'][-1])
                    train_prec = r['server_training_precision'][-1]
                    train_rec = r['server_training_recall'][-1]
                    val_loss = r['server_loss'][-1]
                    val_acc = r['server_accuracy'][-1]
                    val_prec = r['server_precision'][-1]
                    val_rec = r['server_recall'][-1]
                    alg_res = alg_res.append({'algorithm':alg,'alpha':alpha,'run':run,
                        'train_loss':train_loss,'train_acc':train_acc,'train_prec':train_prec,'train_rec':train_rec,
                        'val_loss':val_loss,'val_acc':val_acc,'val_prec':val_prec,'val_rec':val_rec,
                        'rounds':round,'epochs':epoch,'clients_per_round':clients_per_round},ignore_index = True)

                    # Add distribution properties
                    for skew,kurtosis,ks in zip(r['skew'][-1],r['kurtosis'][-1],r['ks_test'][-1]):
                        dist_res = dist_res.append({'algorithm':alg,'alpha':alpha,'run':run,
                            'ks_test':ks,'kurtosis':kurtosis,'skew':skew,'rounds':round,
                            'epochs':epoch,'clients_per_round':clients_per_round},ignore_index = True)
    return alg_res,dist_res



################# Res 1 ###############################################
print('----------- Running test epoch test ------------------------')
rounds = 1
clients_per_rounds = 10
epochs = [1,2,5,10]
alg_res,dist_res = test_script(rounds=rounds,clients_per_rounds=clients_per_rounds,epochs=epochs)
b
alg_res.to_csv(os.path.join(log_path,'alg_res_epochs.csv'))
dist_res.to_csv(os.path.join(log_path,'dist_res_epochs.csv'))

################# Res 2 ###############################################
print('----------- Running n_rounds epoch test ------------------------')
rounds = [1,2,5,10]
clients_per_rounds = 10
epochs = 1
alg_res,dist_res = test_script(rounds=rounds,clients_per_rounds=clients_per_rounds,epochs=epochs)
alg_res.to_csv(os.path.join(log_path,'alg_res_rounds.csv'))
dist_res.to_csv(os.path.join(log_path,'dist_res_rounds.csv'))

################# Res 3 ###############################################
print('----------- Running n_clients epoch test ------------------------')
rounds = 1
clients_per_rounds = [5,10,50,100]
epochs = 1
alg_res,dist_res = test_script(rounds=rounds,clients_per_rounds=clients_per_rounds,epochs=epochs)
alg_res.to_csv(os.path.join(log_path,'alg_res_n_clients.csv'))
dist_res.to_csv(os.path.join(log_path,'dist_res_n_clients.csv'))

#################### Plotting #####################################


#################### Dists #################################
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

######### Res 1 #############################
res = pd.read_csv(os.path.join(result_path,'res_epochs.csv'))
res_plt_path = os.path.join(plot_path,'epochs')
if not os.path.exists(res_plt_path): os.mkdir(res_plt_path)
for metric in metrics:
    plot_func(res,out_path,'epochs',metric)

######### Res 2 #############################
res = pd.read_csv(os.path.join(result_path,'res_rounds.csv'))
res_plt_path = os.path.join(plot_path,'n_rounds')
if not os.path.exists(res_plt_path): os.mkdir(res_plt_path)
for metric in metrics:
    plot_func(res,out_path,'rounds',metric)

######### Res 3 #############################
res = pd.read_csv(os.path.join(result_path,'res_n_clients.csv'))
res_plt_path = os.path.join(plot_path,'epochs')
if not os.path.exists(res_plt_path): os.mkdir(res_plt_path)
for metric in metrics:
    plot_func(res,out_path,'n_clients',metric)
