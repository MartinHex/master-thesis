import sys
sys.path.append('.')
#Import models
from Models.CIFAR_Model import CIFAR_Model as Model
from Dataloaders.DirichletCifar100 import DirichletCifar100 as Dataloader
# Import Algorithms
from Algorithms.FedAvg import FedAvg
from Algorithms.FedKpPa import FedKpPa
from Algorithms.FedPa import FedPa
from Algorithms.FedKp import FedKp
from Algorithms.SGLD import SGLD
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import torch
import numpy as np
import os

seed = 1234
number_of_clients = 500
clients_per_round = 20
batch_size = 20
alpha = 0.1
beta = 10
dataloader = Dataloader(number_of_clients,alpha=alpha,beta=beta,seed=seed)
test_data = dataloader.get_test_dataloader(300)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
print('Creating Callbacks')
cbs = Callbacks(test_data, device = device, verbose = True)
callbacks = [
    cbs.server_loss,
    cbs.server_accuracy,
    cbs.server_training_accuracy,
    cbs.server_training_loss,
    cbs.client_training_accuracy,
    cbs.client_training_loss,
    ]

# Initiate algorithms with same parameters as in papers.

print('Creating FedPA')
fedpa = FedPa(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round, client_lr = 0.01,
    shrinkage = 0.01,batch_size=batch_size,burnin=400,
    server_momentum = 0.9, server_optimizer = 'sgd', server_lr = 0.5)

print('Creating FedAvg')
fedavg = FedAvg(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round,client_lr = 0.01,
    batch_size=batch_size, server_momentum = 0.9, server_lr = 0.5,
    server_optimizer = 'sgd')

print('Creating FedKP')
fedkp_cluster_mean = FedKp(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round, client_lr = 0.01,
    batch_size=batch_size, server_momentum = 0.9, server_lr = 0.5,
    server_optimizer = 'sgd',cluster_mean=True,max_iter=100)

fedkp = FedKp(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round, client_lr = 0.01,
    batch_size=batch_size, server_momentum = 0.9, server_lr = 0.5,
    server_optimizer = 'sgd',cluster_mean=False,max_iter=100)

print('Creating FedKpPa')
fedkppa = FedKpPa(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round, client_lr = 0.01,shrinkage = 0.01,
    batch_size=batch_size, server_momentum = 0.9, server_lr = 0.5,
    server_optimizer = 'sgd',cluster_mean=False,max_iter=100)

print('Creating SGLD')
fedsgld = SGLD(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round,client_lr = 0.01,
    batch_size=batch_size, server_momentum = 0.9, server_lr = 0.5,
    server_optimizer = 'sgd',burn_in=400)

alghs = {
    'FedPA':fedpa,
    'FedAvg':fedavg,
    'FedKp':fedkp,
    'FedKp_cluster_mean':fedkp_cluster_mean,
    'FedKpPa':fedkppa,
    'SGLD':fedsgld
}

print('Setting up save paths')
test_dir = os.path.join('data/Results/CIFAR100')
if not os.path.exists(test_dir): os.mkdir(test_dir)
out_dir = os.path.join(test_dir,'alpha_01_hyper')
if not os.path.exists(out_dir): os.mkdir(out_dir)
model_dir = os.path.join(out_dir,'Models')
if not os.path.exists(model_dir): os.mkdir(model_dir)
log_dir = os.path.join(out_dir,'logs')
if not os.path.exists(log_dir): os.mkdir(log_dir)

# Load initial model and save a new initial model if it doesn't exist.
print('Initializing models')
initial_model_path = os.path.join(model_dir,'initial_model')
if os.path.exists(initial_model_path):
    initial_model = torch.load(initial_model_path)
else:
    initial_model = alghs[list(alghs.keys())[0]].server.get_weights()
    torch.save(initial_model, initial_model_path)

iterations = 1000
print('Running Algorithms')
for alg in alghs:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Running: {}'.format(alg))
    alghs[alg].server.set_weights(initial_model)
    alghs[alg].run(iterations, epochs = 10, device = device,callbacks=callbacks,log_callbacks=True, log_dir = log_dir,file_name=alg)
    out_path = os.path.join(model_dir,'model_%s_iter_%i'%(alg,iterations))
    torch.save(alghs[alg].server.get_weights(), out_path)
