import sys
sys.path.append('.')
#Import models
from Models.MNIST_Model_lite import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
# Import Algorithms
from Algorithms.FedAvg import FedAvg
from Algorithms.FedPa import FedPa
from Algorithms.Algorithm import Algorithm
# Import servers and clients for CustomAlgorithms
from Clients.FedPaClient import FedPaClient
from Servers.FedAvgServer import FedAvgServer
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

number_of_clients = 100
clients_per_round = 20
batch_size = 16
dataloader = Dataloader(number_of_clients)
test_data = dataloader.get_test_dataloader(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
cbs = Callbacks(test_data, device = device, verbose = True)
callbacks = [cbs.server_loss, cbs.server_accuracy]

fedpa = FedPa(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round,client_lr = 0.01,
    batch_size=batch_size, shrinkage = 0.1, burnin = 5)

fedavg = FedAvg(dataloader=dataloader, Model=Model,
    clients_per_round = clients_per_round,client_lr = 0.01,
    batch_size=batch_size)

# Initiate algorithms with same parameters as in papers.
alghs = {
    'FedAvg': fedavg,
    'FedPA': fedpa,

}

print('Setting up save paths')
test_dir = os.path.join('data/results/MNIST')
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

iterations = 10
print('Running Algorithms')
for alg in alghs:
    torch.manual_seed(0)
    np.random.seed(0)
    print('Running: {}'.format(alg))
    alghs[alg].server.set_weights(initial_model)
    alghs[alg].run(iterations, epochs = 5, device = device,callbacks=callbacks,log_callbacks=True, log_dir = log_dir,file_name=alg)
    out_path = os.path.join(model_dir,'model_%s_iter_%i'%(alg,iterations))
    torch.save(alghs[alg].server.get_weights(), out_path)
