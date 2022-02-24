from Servers.FedKPServer import FedKPServer
from Clients.SGDClient import SGDClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedAvg(ABCAlgorithm):
    def __init__(self,dataloader,Model,callbacks=None,n_clients=5, save_callbacks = False):
        super().__init__(n_clients,dataloader,Model, callbacks, save_callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(n_clients)
        self.clients = [FedKPServer(Model(), loader) for loader in client_dataloaders]
        self.server = FedKPServer(Model())
