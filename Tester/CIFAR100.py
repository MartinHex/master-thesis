import sys
sys.path.append('.')
#Import models
from Models.CIFAR_Model import CIFAR_Model as Model
from Dataloaders.DirichletCifar100 import DirichletCifar100 as Dataloader
# Import Algorithms
from Algorithms.FedAvg import FedAvg
from Algorithms.FedBe import FedBe
from Algorithms.FedPa import FedPa
from Algorithms.FedKp import FedKp
from Algorithms.FedAg import FedAg
from Algorithms.Algorithm import Algorithm
# Import servers and clients for CustomAlgorithms
from Clients.FedPaClient import FedPaClient
from Servers.FedAvgServer import FedAvgServer
from Clients.SGDClient import SGDClient
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import matplotlib.pyplot as plt
import torch


number_of_clients = 500
clients_per_round = 20
batch_size = 16
alpha = 0.1
beta = 100
dataloader = Dataloader(number_of_clients,alpha=alpha,beta=beta)
test_data = dataloader.get_test_dataloader(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
print('Creating Callbacks')
cbs_pa = Callbacks(test_data, device = device, verbose = True)
cbs_be = Callbacks(test_data, device = device, verbose = True)
cbs_ag = Callbacks(test_data, device = device, verbose = True)
cbs_kp = Callbacks(test_data, device = device, verbose = True)
callbacks_pa = [
    ('server_pa_loss', cbs_pa.server_loss),
    ('server_pa_accuracy', cbs_pa.server_accuracy),
]
callbacks_be = [
    ('server_be_loss', cbs_be.server_loss),
    ('server_be_accuracy', cbs_be.server_accuracy),
]
callbacks_ag = [
    ('server_ag_loss', cbs_ag.server_loss),
    ('server_ag_accuracy', cbs_ag.server_accuracy),
]
callbacks_kp = [
    ('server_kp_loss', cbs_kp.server_loss),
    ('server_kp_accuracy', cbs_kp.server_accuracy),
]

# Initiate algorithms with same parameters as in papers.
# Set parameters to replicate paper results
print('Creating FedAG')
fedag = FedAg(dataloader=dataloader, Model=Model, callbacks = callbacks_ag,
    clients_per_round = clients_per_round, save_callbacks = True, batch_size = batch_size)

print('Creating FedBE')
fedbe = FedBe(dataloader=dataloader, Model=Model, callbacks = callbacks_be,
    clients_per_round = clients_per_round, save_callbacks = True, batch_size = batch_size)

print('Creating FedPA')
fedpa = FedPa(dataloader=dataloader, Model=Model, callbacks = callbacks_pa,
    clients_per_round = clients_per_round, save_callbacks = True,client_lr = 0.01,
    burn_in =  0.8, shrinkage = 0.01,batch_size=batch_size)

print('Creating FedKP')
fedkp = FedKp(dataloader=dataloader, Model=Model, callbacks = callbacks_kp,
    clients_per_round = clients_per_round, save_callbacks = True,
    cov_adj=False,batch_size = batch_size)

alghs = [
    fedbe,
    fedag,
    fedpa,
    fedkp
]

print('Initializing Clients')
alghs[0].server.push_weights([alg.server for alg in alghs[1:]])
iterations = 600
print('Running Algorithms')
for alg in alghs:
    alg.run(iterations, epochs = 5, device = device)
