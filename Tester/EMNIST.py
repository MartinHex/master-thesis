import sys
sys.path.append('.')
#Import models
from Models.EMNIST_Model import EMNIST_Model as Model
from Dataloaders.Emnist62 import EMNIST as Dataloader
# Import Algorithms
from Algorithms.FedAvg import FedAvg
from Algorithms.FedBe import FedBe
from Algorithms.FedPa import FedPa
from Algorithms.FedKp import FedKp
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import torch
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

# Initiate algorithms with same parameters as in papers.
# Set parameters to replicate paper results
print('Creating FedBE')
fedbe = FedBe(dataloader=dataloader, Model=Model ,client_lr = 0.1,
    clients_per_round = clients_per_round, batch_size = batch_size,
    p_validation=0.01)

print('Creating FedPA')
fedpa = FedPa(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round, client_lr = 0.1,
    shrinkage = 0.01,batch_size=batch_size,burnin=100)

print('Creating FedAvg')
fedavg = FedAvg(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round,client_lr = 0.1,
    batch_size=batch_size)

print('Creating FedKP')
fedkp = FedKp(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round, client_lr = 0.1,
    batch_size = batch_size,cluster_mean=True,max_iter=20)

alghs = {
    'FedAvg_burnin':fedavg,
    'FedKP':fedkp,
    'FedPA':fedpa,
    'FedBe':fedbe,
}

print('Setting up save paths')
test_dir = os.path.join('data/Results/EMNIST')
if not os.path.exists(test_dir): os.mkdir(test_dir)
out_dir = os.path.join(test_dir,'experiment')
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

iterations = 100
print('Running Algorithms')
for alg in alghs:
    print('Running: {}'.format(alg))
    alghs[alg].server.set_weights(initial_model)
    alghs[alg].run(iterations, epochs = 5, device = device,callbacks=callbacks,log_callbacks=True, log_dir = log_dir,file_name=alg)
    out_path = os.path.join(model_dir,'model_%s_iter_%i'%(alg,iterations))
    torch.save(alghs[alg].server.get_weights(), out_path)
