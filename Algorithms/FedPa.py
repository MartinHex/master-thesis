from Servers.FedAvgServer import FedAvgServer
from Clients.FedPaClient import FedPaClient
from Algorithms.ABCAlgorithm import ABCAlgorithm

class FedPa(ABCAlgorithm):
    def __init__(self,dataloader,Model,callbacks=None,n_clients=5):
        super().__init__(n_clients,dataloader,Model, callbacks)
        client_dataloaders = dataloader.get_training_dataloaders(n_clients)
        self.clients = [FedPaClient(Model(), loader) for loader in client_dataloaders]
        self.server = FedAvgServer(Model())
