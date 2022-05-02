import sys
sys.path.append('.')
#Import models
from Models.MNIST_Model_lite import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
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

number_of_clients = 100
clients_per_round = 20
batch_size = 16
alpha=1
dataloader = Dataloader(number_of_clients)
test_data = dataloader.get_test_dataloader(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

server_momentum = 0.
client_momentum = 0.
server_lr = 1
client_lr = 0.01
burn_in = 10
shrinkage = 0.1
max_iter = 20
server_optimizer = 'none'
bandwidth = 'silverman'
kernel_function = 'epanachnikov'
iterations = 20
local_epochs = 1
seed = 0

# Create callback functions that are run at the end of every round
cbs = Callbacks(test_data, device = device, verbose = True)
callbacks = [cbs.server_loss, cbs.server_accuracy]

# Create callback functions that are run at the end of every round
print('Creating Callbacks')
cbs = Callbacks(test_data, device = device, verbose = False)
callbacks = [
    cbs.server_thesis_results,
    cbs.server_training_thesis_results,
    ]

# Initiate algorithms with same parameters as in papers.
# Set parameters to replicate paper results
print('Creating FedPA')
torch.manual_seed(seed)
np.random.seed(seed)
fedpa = FedPa(
        dataloader=dataloader,
        Model=Model,
        clients_per_round = clients_per_round,
        client_lr = client_lr,
        shrinkage = shrinkage,
        batch_size = batch_size,
        burnin = burn_in,
        server_momentum = server_momentum,
        server_optimizer = server_optimizer,
        server_lr = server_lr,
        momentum = client_momentum,
        clients_sample_alpha = alpha,
    )

print('Creating FedAvg')
torch.manual_seed(seed)
np.random.seed(seed)
fedavg = FedAvg(
        dataloader=dataloader,
        Model=Model,
        clients_per_round = clients_per_round,
        client_lr = client_lr,
        batch_size = batch_size,
        server_momentum = server_momentum,
        server_optimizer = server_optimizer,
        server_lr = server_lr,
        momentum = client_momentum,
        clients_sample_alpha = alpha,
    )

print('Creating FedKP')
torch.manual_seed(seed)
np.random.seed(seed)
fedkp_cluster_mean = FedKp(
        dataloader=dataloader,
        Model=Model,
        clients_per_round = clients_per_round,
        client_lr = client_lr,
        batch_size = batch_size,
        server_momentum = server_momentum,
        server_optimizer = server_optimizer,
        server_lr = server_lr,
        momentum = client_momentum,
        cluster_mean = True,
        max_iter = max_iter,
        clients_sample_alpha = alpha,
        bandwidth = bandwidth,
        kernel_function = kernel_function,
    )

torch.manual_seed(seed)
np.random.seed(seed)
fedkp = FedKp(
        dataloader=dataloader,
        Model=Model,
        clients_per_round = clients_per_round,
        client_lr = client_lr,
        batch_size = batch_size,
        server_momentum = server_momentum,
        server_optimizer = server_optimizer,
        server_lr = server_lr,
        momentum = client_momentum,
        cluster_mean = False,
        max_iter = 100,
        clients_sample_alpha = alpha,
        bandwidth = bandwidth,
        kernel_function = kernel_function,
    )

print('Creating FedKpPa')
torch.manual_seed(seed)
np.random.seed(seed)
fedkppa = FedKpPa(
        dataloader=dataloader,
        Model=Model,
        clients_per_round = clients_per_round,
        client_lr = client_lr,
        shrinkage = shrinkage,
        batch_size = batch_size,
        server_momentum = server_momentum,
        server_optimizer = server_optimizer,
        server_lr = server_lr,
        momentum = client_momentum,
        cluster_mean = True,
        max_iter = 100,
        burnin = burn_in,
        clients_sample_alpha = alpha,
        bandwidth = bandwidth,
        kernel_function = kernel_function,
    )

print('Creating SGLD')
torch.manual_seed(seed)
np.random.seed(seed)
fedsgld = SGLD(
        dataloader=dataloader,
        Model=Model,
        clients_per_round = clients_per_round,
        client_lr = client_lr,
        batch_size = batch_size,
        burn_in = burn_in,
        server_momentum = server_momentum,
        server_optimizer = server_optimizer,
        server_lr = server_lr,
        momentum = client_momentum,
        clients_sample_alpha = alpha,
    )

alghs = {
    'FedPA':fedpa,
    'FedAvg':fedavg,
    'FedKP_cluter_mean':fedkp_cluster_mean,
    'FedKP':fedkp,
    'FedKPPA':fedkppa,
    'SGLD':fedsgld
}

print('Setting up save paths')
test_dir = os.path.join('data/Results/MNIST')
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
