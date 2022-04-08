import sys
sys.path.append('.')
#Import models
from Models.MNIST_Model_lite import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
# Import Algorithms
from Algorithms.FedAvg import FedAvg
# from Algorithms.FedBe import FedBe
from Algorithms.SGLD import SGLD
# from Algorithms.FedKp import FedKp
# from Algorithms.FedAg import FedAg
# from Algorithms.Algorithm import Algorithm
# Import servers and clients for CustomAlgorithms
# from Clients.FedPaClient import FedPaClient
# from Servers.FedAvgServer import FedAvgServer
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

number_of_clients = 200
clients_per_round = 10
batch_size = 64
dataloader = Dataloader(number_of_clients, alpha = 0.1)
test_data = dataloader.get_test_dataloader(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
cbs = Callbacks(test_data, device = device, verbose = True)
callbacks = [cbs.server_accuracy, cbs.server_loss]

# print('Creating FedPA')
sgld = SGLD(dataloader=dataloader, Model=Model, clients_per_round = clients_per_round,
    client_lr = 0.01, batch_size=batch_size, burn_in=10, server_momentum = 0.9,
    server_optimizer = 'sgd', server_lr = 0.5, momentum = 0.9)

print('Creating FedAvg')
fedavg = FedAvg(dataloader=dataloader, Model=Model, clients_per_round = clients_per_round,
    client_lr = 0.01, batch_size=batch_size)#, server_momentum = 0.9,
    # server_optimizer = 'sgd', server_lr = 0.5, momentum = 0.5)

alghs = {
    'SGLD': sgld,
    'FedAvg': fedavg,
}
print('Initializing Clients')
cwd = os.getcwd()
model_base_dir = os.path.join(cwd, 'data', 'results','MNIST', 'SGLD', 'models')
logs_base_dir = os.path.join(cwd, 'data', 'results','MNIST', 'SGLD', 'logs')
initial_model_path = os.path.join(model_base_dir, 'initial_model')

initial_model_path = os.path.join(model_base_dir,'initial_model')
if os.path.exists(initial_model_path):
    initial_model = torch.load(initial_model_path)
else:
    initial_model = alghs[list(alghs.keys())[0]].server.get_weights()
    torch.save(initial_model, initial_model_path)

iterations = 50
for name in alghs:
    torch.manual_seed(0)
    np.random.seed(0)
    print('Running: {}'.format(name))
    alghs[name].run(iterations, epochs = 5, callbacks = callbacks, log_callbacks = True,
        log_dir = logs_base_dir, file_name = name)
    model_dir = os.path.join(model_base_dir, name)
    torch.save(alghs[name].server.get_weights(), model_dir)
