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

# General parameters
iterations = 800
seed = 0
alpha = 10
beta = 10
# Dataloader hyperparameters
local_epochs = 20
number_of_clients = 500
clients_per_round = 20
batch_size = 10
# Training hyperparameters
client_lr = 0.01
server_lr = 0.5
server_optimizer = 'sgd'
server_momentum = 0.9
momentum = 0.9
shrinkage = 0.01
burnin = 400
bandwidth = 'silverman'
kernel_function = 'epanachnikov'


dataloader = Dataloader(number_of_clients,alpha=alpha,beta=beta,seed=seed)
test_data = dataloader.get_test_dataloader(300)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
print('Creating Callbacks')
cbs = Callbacks(test_data, device = device, verbose = False)
callbacks = [
    cbs.server_thesis_results,
    cbs.server_training_thesis_results,
    ]

print('Creating FedAvg')
torch.manual_seed(seed)
np.random.seed(seed)
fedavg = FedAvg(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
     momentum = momentum,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_lr = server_lr,
    server_optimizer = server_optimizer)

print('Creating FedPA')
torch.manual_seed(seed)
np.random.seed(seed)
fedpa= FedPa(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    shrinkage = shrinkage,
    batch_size=batch_size,
    burnin=burnin,
    server_momentum = server_momentum,
    server_optimizer = server_optimizer,
    server_lr = server_lr,
    momentum = momentum)

print('Creating FedKP')
torch.manual_seed(seed)
np.random.seed(seed)
fedkp_cluster_mean = FedKp(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    momentum = momentum,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_lr = server_lr,
    server_optimizer = server_optimizer,
    cluster_mean=True,
    kernel_function = kernel_function,
    bandwidth = bandwidth,)

torch.manual_seed(seed)
np.random.seed(seed)
fedkp = FedKp(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    momentum = momentum,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_lr = server_lr,
    server_optimizer = server_optimizer,
    cluster_mean=False,
    kernel_function = kernel_function,
    bandwidth = bandwidth,)

print('Creating FedKpPa')
torch.manual_seed(seed)
np.random.seed(seed)
fedkppa = FedKpPa(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    shrinkage = shrinkage,
    momentum = momentum,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_lr = server_lr,
    server_optimizer = server_optimizer,
    cluster_mean=True,
    kernel_function = kernel_function,
    bandwidth = bandwidth,
    burnin=burnin)

print('Creating SGLD')
torch.manual_seed(seed)
np.random.seed(seed)
fedsgld = SGLD(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    momentum = momentum,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_lr = server_lr,
    server_optimizer = server_optimizer,
    burn_in=burnin)

alghs = {
    'FedAvg':fedavg,
    'FedPa':fedpa,
    'FedKp':fedkp,
    'FedKp_cluster_mean':fedkp_cluster_mean,
    'FedKpPa':fedkppa,
    'SGLD':fedsgld,
}

print('Setting up save paths')
test_dir = os.path.join('data/Results/CIFAR100')
if not os.path.exists(test_dir): os.mkdir(test_dir)
out_dir = os.path.join(test_dir,'alpha_{}'.format(str(alpha).replace('.', '')))
if not os.path.exists(out_dir): os.mkdir(out_dir)
model_dir = os.path.join(out_dir,'Models')
if not os.path.exists(model_dir): os.mkdir(model_dir)
log_dir = os.path.join(out_dir,'logs')
if not os.path.exists(log_dir): os.mkdir(log_dir)

# Load initial model and save a new initial model if it doesn't exist.
print('Initializing models')
initial_model_path = os.path.join(test_dir,'initial_model')
if os.path.exists(initial_model_path):
    initial_model = torch.load(initial_model_path)
else:
    torch.manual_seed(seed)
    np.random.seed(seed)
    initial_model = alghs[list(alghs.keys())[0]].server.get_weights()
    torch.save(initial_model, initial_model_path)

print('Running Algorithms')
for alg in alghs:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Running: {}'.format(alg))
    alghs[alg].server.set_weights(initial_model)
    alghs[alg].run(
        iterations,
        epochs = local_epochs,
        device = device,
        callbacks=callbacks,
        log_callbacks=True,
        log_dir = log_dir,
        file_name=alg
    )
    out_path = os.path.join(model_dir,'model_%s'%(alg))
    torch.save(alghs[alg].server.get_weights(), out_path)
    # Free up memory
    del alg
