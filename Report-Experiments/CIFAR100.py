import sys
sys.path.append('.')
#Import models
from Models.CIFAR_Model import CIFAR_Model as Model
from Dataloaders.CIFAR100 import CIFAR100 as Dataloader
# Import Algorithms
from Algorithms.FedAvg import FedAvg
from Algorithms.FedPa import FedPa
from Algorithms.FedProx import FedProx
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import torch
import numpy as np
import os

# General parameters
iterations = 1000
seed = 0
alpha = 10
# Dataloader hyperparameters
local_epochs = 20
number_of_clients = 500
clients_per_round = 20
batch_size = 10
# Training hyperparameters
server_optimizer = 'sgd'
server_momentum = 0.0
momentum = 0.9
client_lr = 10**(-1)
server_lr = 10**(1/2)

# Mean shifts used
meanshifts = ['','client-shift','mean-shift']


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

print('Creating FedAvg')
torch.manual_seed(seed)
np.random.seed(seed)
fedavg = FedAvg(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    server_lr = server_lr,
    momentum = momentum,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_optimizer = server_optimizer
    )

print('Creating FedAdam')
torch.manual_seed(seed)
np.random.seed(seed)
fedadam = FedAvg(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = 10**(-3/2),
    server_lr = 10**(0),
    momentum = momentum,
    batch_size=batch_size,
    server_optimizer = 'adam',
    b1=0.9,
    b2=0.99,
    tau=0.1
    )

print('Creating FedYoGi')
torch.manual_seed(seed)
np.random.seed(seed)
fedyogi = FedAvg(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = 10**(-3/2),
    server_lr = 10**(0),
    momentum = momentum,
    batch_size=batch_size,
    server_optimizer = 'yogi',
    b1=0.9,
    b2=0.99,
    tau=0.1
    )

print('Creating FedProx')
torch.manual_seed(seed)
np.random.seed(seed)
fedprox = FedProx(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    server_lr = server_lr,
    momentum = momentum,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_optimizer = server_optimizer,
    mu = 0.01)

print('Creating FedPA')
torch.manual_seed(seed)
np.random.seed(seed)
fedpa= FedPa(
    dataloader=dataloader,
    Model=Model,
    clients_per_round = clients_per_round,
    client_lr = client_lr,
    server_lr = server_lr,
    shrinkage = 0.01,
    burnin = 400,
    batch_size=batch_size,
    server_momentum = server_momentum,
    server_optimizer = 'sgd',
    momentum = momentum)

alghs = {
    'FedAvg':fedavg,
    'FedAdam':fedadam,
    'FedYoGi':fedyogi,
    'FedProx': fedprox,
    'FedPA':fedpa
}

print('Setting up save paths')
test_dir = os.path.join('data/ReportResults/CIFAR10')
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
    for meanshift in meanshifts:
        alghs[alg].server.meanshift = meanshift
        name = meanshift+'_'+alg if meanshift!='' else alg
        torch.manual_seed(seed)
        np.random.seed(seed)
        print('Running: {}'.format(name))
        alghs[alg].server.set_weights(initial_model)
        alghs[alg].run(
            iterations,
            epochs = local_epochs,
            device = device,
            callbacks=callbacks,
            log_callbacks=True,
            log_dir = log_dir,
            file_name=name
        )
        out_path = os.path.join(model_dir,'model_%s'%(name))
        torch.save(alghs[alg].server.get_weights(), out_path)
    # Free up memory
    del alg
    torch.cuda.empty_cache()
