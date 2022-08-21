import sys
sys.path.append('.')
#Import models
from Models.CIFAR_Model import CIFAR_Model as Model
from Dataloaders.CIFAR10 import CIFAR10 as Dataloader
# Import Algorithms
from Algorithms.FedAvg import FedAvg
from Algorithms.FedKp import FedKp
from Algorithms.FedProx import FedProx
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import torch
import numpy as np
import os

# General parameters
iterations = 100
seed = 0
alpha = 10
# Dataloader hyperparameters
local_epochs = 10
number_of_clients = 500
clients_per_round = 20
batch_size = 20
# Training hyperparameters
server_optimizers = ['sgd','adam','yogi']
server_momentum = 0.0
momentum = 0.9
client_lr = 10**(-1/2)
server_lr = 10**(-1/2)
shrinkage = 0.01
burnin = 400
mu = 0.01
bandwidth = 'silverman'
kernel_function = 'epanachnikov'
b1=0.9
b2=0.99
tau=0.1

dataloader = Dataloader(number_of_clients,alpha=alpha,seed=seed)
test_data = dataloader.get_test_dataloader(300)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
print('Creating Callbacks')
cbs = Callbacks(test_data, device = device, verbose = False)
callbacks = [
    cbs.server_thesis_results,
    cbs.server_training_thesis_results,
    ]

print('Setting up save paths')
test_dir = os.path.join('data/Results/CIFAR10')
if not os.path.exists(test_dir): os.mkdir(test_dir)
out_dir = os.path.join(test_dir,'alpha_{}'.format(str(alpha).replace('.', '')))
if not os.path.exists(out_dir): os.mkdir(out_dir)
model_dir = os.path.join(out_dir,'Models')
if not os.path.exists(model_dir): os.mkdir(model_dir)
log_dir = os.path.join(out_dir,'logs')
if not os.path.exists(log_dir): os.mkdir(log_dir)

# Run experiment
for server_optimizer in server_optimizers:
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    fedprox = FedProx(
        dataloader=dataloader,
        Model=Model,
        clients_per_round = clients_per_round,
        client_lr = client_lr,
         momentum = momentum,
        batch_size=batch_size,
        server_momentum = server_momentum,
        server_lr = server_lr,
        server_optimizer = server_optimizer,
        mu = mu,
        b1=b1,
        b2=b2,
        tau=tau)

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
        bandwidth = bandwidth,
        b1=b1,
        b2=b2,
        tau=tau)

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
        bandwidth = bandwidth,
        b1=b1,
        b2=b2,
        tau=tau)

    alghs = {
        'FedAvg':fedavg,
        'FedKp':fedkp,
        'FedKp_cluster_mean':fedkp_cluster_mean,
        'FedProx': fedprox,
    }

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
        namme = alg+'_'+server_optimizer
        torch.manual_seed(seed)
        np.random.seed(seed)
        print('Running: {}'.format(namme))
        alghs[alg].server.set_weights(initial_model)
        alghs[alg].run(
            iterations,
            epochs = local_epochs,
            device = device,
            callbacks=callbacks,
            log_callbacks=True,
            log_dir = log_dir,
            file_name=alg+namme
        )
        out_path = os.path.join(model_dir,'model_%s'%(namme))
        torch.save(alghs[alg].server.get_weights(), out_path)
        # Free up memory
        del alg