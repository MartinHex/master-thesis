#Import models
from Models.MNIST_Model_lite import MNIST_Model as Model
from Dataloaders.Mnist import Mnist as Dataloader
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

number_of_clients = 100
batch_size = 16
dataloader = Dataloader(number_of_clients)
test_data = dataloader.get_test_dataloader(batch_size)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

# Create callback functions that are run at the end of every round
cbs_pa = Callbacks(test_data, device = device, verbose = False)
cbs_be = Callbacks(test_data, device = device, verbose = False)
cbs_ag = Callbacks(test_data, device = device, verbose = False)
cbs_kp = Callbacks(test_data, device = device, verbose = False)
callbacks_pa = [
    ('client_loss', cbs_pa.client_loss),
    ('server_pa_loss', cbs_pa.server_loss),
    ('client_accuracy', cbs_pa.client_accuracy),
    ('server_pa_accuracy', cbs_pa.server_accuracy),
]
callbacks_be = [
    ('client_loss', cbs_be.client_loss),
    ('server_be_loss', cbs_be.server_loss),
    ('client_accuracy', cbs_be.client_accuracy),
    ('server_be_accuracy', cbs_be.server_accuracy),
]
callbacks_ag = [
    ('client_loss', cbs_ag.client_loss),
    ('server_ag_loss', cbs_ag.server_loss),
    ('client_accuracy', cbs_ag.client_accuracy),
    ('server_ag_accuracy', cbs_ag.server_accuracy),
]
callbacks_kp = [
    ('client_loss', cbs_kp.client_loss),
    ('server_kp_loss', cbs_kp.server_loss),
    ('client_accuracy', cbs_kp.client_accuracy),
    ('server_kp_accuracy', cbs_kp.server_accuracy),
]

# Set parameters to replicate paper results
fedpa_clients = [FedPaClient( Model(), dl, learning_rate = 0.01, burn_in =  100,
                                K = 5, shrinkage = 0.1, mcmc_samples = 100)
                                for dl in dataloader.get_training_dataloaders(16)]
fedpa_server = FedAvgServer(Model())

fedpa = Algorithm(fedpa_server,fedpa_clients, callbacks = callbacks_pa)
# Initiate algorithms with same parameters as in papers.
alghs = [FedAg(dataloader=dataloader, Model=Model, callbacks = callbacks_ag, save_callbacks = True),
        FedBe(dataloader=dataloader, Model=Model, callbacks = callbacks_be, save_callbacks = True),
        fedpa,
        FedKp(dataloader=dataloader, Model=Model, callbacks = callbacks_kp, save_callbacks = True)
]

alghs[0].server.push_weights([alg.server for alg in alghs[1:]])
iterations = 30
for alg in alghs:
    alg.run(iterations, epochs = 5)
