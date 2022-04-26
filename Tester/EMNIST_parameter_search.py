import sys
sys.path.append('.')
#Import models
from Models.EMNIST_Model import EMNIST_Model as Model
from Dataloaders.Emnist62 import EMNIST as Dataloader
# Import Algorithms
from Algorithms.FedPa import FedPa
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import torch
import numpy as np
import os

clients_per_round = 100
batch_size = 20
dataloader = Dataloader()
test_data = dataloader.get_test_dataloader(300)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
print('Creating Callbacks')
cbs = Callbacks(test_data, device = device, verbose = True)
callbacks = [
    cbs.server_loss,
    cbs.server_accuracy,
    ]

server_momentum = 0.9
client_momentum = 0.9
server_lr = 0.5
client_lr = 0.01
server_optimizer = 'sgd'
burn_ins = [600, 500, 400]
shrinkages = [1, 0.5, 0.1]
iterations = 1000
local_epochs = 5

print('Setting up save paths')
test_dir = os.path.join('data/Results/EMNIST')
if not os.path.exists(test_dir): os.mkdir(test_dir)
out_dir = os.path.join(test_dir,'hyperparameter_search')
if not os.path.exists(out_dir): os.mkdir(out_dir)
model_dir = os.path.join(out_dir,'Models')
if not os.path.exists(model_dir): os.mkdir(model_dir)
log_dir = os.path.join(out_dir,'logs')
if not os.path.exists(log_dir): os.mkdir(log_dir)

for burn_in in burn_ins:
    for shrinkage in shrinkages:
        print('Creating FedPA with shrinkage: {} and burn in: {}'.format(shrinkage, burn_in))
        model_name = 'FedPA_rho_{}_burnin_{}'.format(shrinkage, burn_in)
        torch.manual_seed(0)
        np.random.seed(0)
        fedpa = FedPa(
                dataloader=dataloader,
                Model=Model,
                clients_per_round = clients_per_round,
                client_lr = client_lr,
                shrinkage = shrinkage,
                batch_size=batch_size,
                burnin=burn_in,
                server_momentum = server_momentum,
                server_optimizer = server_optimizer,
                server_lr = server_lr,
                momentum = client_momentum,
            )
        torch.manual_seed(0)
        np.random.seed(0)
        fedpa.run(
            iterations,
            epochs = local_epochs,
            device = device,
            callbacks = callbacks,
            log_callbacks = True,
            log_dir = log_dir,
            file_name = model_name
        )
        out_path = os.path.join(model_dir,model_name)
        torch.save(fedpa.server.get_weights(), out_path)
