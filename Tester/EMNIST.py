#Import models
from Models.EMNIST_Model import EMNIST_Model as Model
from Dataloaders.Emnist62 import EMNIST as Dataloader
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
# Additional imports
from Models.Callbacks.Callbacks import Callbacks
import matplotlib.pyplot as plt
import torch

number_of_clients = 5
batch_size = 16
dataloader = Dataloader(number_of_clients)
test_data = dataloader.get_test_dataloader(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
cbs = Callbacks(test_data, device = device)
callbacks = [
    ('client_loss', cbs.client_loss),
    ('server_loss', cbs.server_loss),
    ('client_accuracy', cbs.client_accuracy),
    ('server_accuracy', cbs.server_accuracy),
]

# Set parameters to replicate paper results
fedpa_clients = [FedPaClient( Model(), dl, learning_rate = 0.01, burn_in =  100,
                                K = 5, shrinkage = 0.1, mcmc_samples = 100)
                                for dl in dataloader.get_training_dataloaders(16)]
fedpa_server = FedAvgServer(Model())

fedpa = Algorithm(fedpa_server,fedpa_clients)

FedBe(dataloader=dataloader, Model=Model, callbacks = callbacks, save_callbacks = True)
# Initiate algorithms with same parameters as in papers.
alghs = [FedAg(dataloader=dataloader, Model=Model, callbacks = callbacks, save_callbacks = True),
        FedBe(dataloader=dataloader, Model=Model, callbacks = callbacks, save_callbacks = True),
        fedpa,
        FedKp(dataloader=dataloader, Model=Model, callbacks = callbacks, save_callbacks = True)
]

alghs[0].server.push_weights([alg.server for alg in alghs[1:]])
iterations = 30
for alg in alghs:
    alg.run(iterations)
